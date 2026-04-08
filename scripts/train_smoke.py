"""Smoke training run — validates the full training pipeline with real data.

Runs 2000 steps at 1024px with batch 4. No wandb, just console logging.
"""

import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reswapper.models.generator import FaceSwapGenerator
from reswapper.models.discriminator import (
    ProjectedDiscriminator, rpgan_loss_d, rpgan_loss_g, r1_penalty, r2_penalty,
)
from reswapper.models.vae import FluxVAE
from reswapper.models.face_parser import FaceParser
from reswapper.models.ema import EMAModel
from reswapper.losses.reconstruction import ReconstructionLoss
from reswapper.losses.occlusion import OcclusionMaskLoss
from reswapper.data.dataset import VGGFace2Dataset
from reswapper.data.augmentations import FaceSwapAugmentation
from reswapper.training.scheduler import cosine_warmup_scheduler


def main():
    device = torch.device("cuda")
    BATCH = 4
    TOTAL_STEPS = 2000
    LOG_EVERY = 50
    SAVE_EVERY = 500

    # Models
    gen = FaceSwapGenerator(
        latent_channels=16, base_channels=256, bottleneck_channels=512
    ).to(device)
    disc = ProjectedDiscriminator(backbone="dinov2_vitb14").to(device)
    vae = FluxVAE(pretrained_path="black-forest-labs/FLUX.1-schnell")
    vae.load(device, dtype=torch.bfloat16)
    ema = EMAModel(gen, decay=0.9999)

    # Optimizers
    opt_g = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.0, 0.99))
    opt_d = torch.optim.Adam(
        [p for p in disc.parameters() if p.requires_grad], lr=4e-4, betas=(0.0, 0.99)
    )
    sched_g = cosine_warmup_scheduler(opt_g, 200, TOTAL_STEPS)
    sched_d = cosine_warmup_scheduler(opt_d, 200, TOTAL_STEPS)

    # Data
    aug = FaceSwapAugmentation()
    ds = VGGFace2Dataset(
        dataset_dir="data/preprocessed_vggface2",
        resolution=1024,
        same_identity_ratio=0.5,
        transform=aug,
    )
    dl = DataLoader(
        ds, batch_size=BATCH, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True,
    )

    occ_loss_fn = OcclusionMaskLoss()

    os.makedirs("checkpoints", exist_ok=True)

    gen_params = sum(p.numel() for p in gen.parameters()) / 1e6
    disc_params = sum(p.numel() for p in disc.parameters() if p.requires_grad) / 1e6
    print(f"Training: {len(ds)} samples, batch={BATCH}, steps={TOTAL_STEPS}")
    print(f"Gen: {gen_params:.1f}M, Disc: {disc_params:.1f}M trainable")
    print()

    gen.train()
    disc.train()
    data_iter = iter(dl)
    t0 = time.time()

    for step in range(TOTAL_STEPS):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)

        target_image = batch["target_image"].to(device)
        source_emb = batch["source_embedding"].to(device)
        is_same = batch["is_same_identity"].to(device)

        # VAE encode
        target_latent = vae.encode(target_image)
        lh, lw = target_latent.shape[2], target_latent.shape[3]
        parsing = torch.zeros(BATCH, 19, lh, lw, device=device)
        parsing[:, 1] = 1.0  # placeholder: all skin

        # --- Generator forward ---
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            swapped_latent, swap_mask = gen(target_latent.float(), source_emb, parsing)
        output_latent = swap_mask * swapped_latent + (1 - swap_mask) * target_latent.float()
        output_pixels = vae.decode(output_latent)

        # --- D step ---
        opt_d.zero_grad()
        real_logits = disc(target_image)
        fake_logits = disc(output_pixels.detach())
        d_loss = rpgan_loss_d(real_logits, fake_logits)
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in disc.parameters() if p.requires_grad], 1.0
        )
        opt_d.step()

        # R1+R2 every 16 steps
        if step % 16 == 0:
            r1 = r1_penalty(disc, target_image.detach()) * 5.0
            r2 = r2_penalty(disc, output_pixels.detach()) * 5.0
            opt_d.zero_grad()
            (r1 + r2).backward()
            opt_d.step()

        # --- G step ---
        opt_g.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            swapped_latent, swap_mask = gen(target_latent.float(), source_emb, parsing)
        output_latent = swap_mask * swapped_latent + (1 - swap_mask) * target_latent.float()
        output_pixels = vae.decode(output_latent)

        fake_logits_g = disc(output_pixels)
        real_logits_g = disc(target_image.detach())
        g_adv = rpgan_loss_g(real_logits_g, fake_logits_g)

        g_total = g_adv
        g_rec = torch.tensor(0.0)
        if is_same.any():
            g_rec = F.l1_loss(output_pixels[is_same], target_image[is_same]) * 10.0
            g_total = g_total + g_rec

        # Mask losses
        mask_losses = occ_loss_fn(swap_mask)
        for v in mask_losses.values():
            g_total = g_total + v

        g_total.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        opt_g.step()
        sched_g.step()
        sched_d.step()

        if step >= 100:
            ema.update(gen)

        # Logging
        if step % LOG_EVERY == 0:
            elapsed = time.time() - t0
            ms = elapsed / (step + 1) * 1000
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(
                f"[{step}/{TOTAL_STEPS}] "
                f"d={d_loss.item():.4f} g={g_total.item():.4f} "
                f"rec={g_rec.item():.4f} mask={swap_mask.mean().item():.3f} "
                f"mem={mem:.1f}GB {ms:.0f}ms/step",
                flush=True,
            )

        # Checkpoint
        if step > 0 and step % SAVE_EVERY == 0:
            path = f"checkpoints/checkpoint-{step}.pt"
            torch.save(
                {
                    "step": step,
                    "generator": gen.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer_g": opt_g.state_dict(),
                    "optimizer_d": opt_d.state_dict(),
                },
                path,
            )
            print(f"Saved {path}", flush=True)

    # Final save
    torch.save(
        {"step": TOTAL_STEPS, "generator": gen.state_dict(), "ema": ema.state_dict()},
        f"checkpoints/checkpoint-{TOTAL_STEPS}.pt",
    )
    total_time = time.time() - t0
    print(f"\nTraining complete! {TOTAL_STEPS} steps in {total_time/60:.1f}min")


if __name__ == "__main__":
    main()
