"""Full training run for ReSwapper v2 at 1024px.

Includes all losses: adversarial, reconstruction, identity, LPIPS, occlusion.
Logs to W&B. Saves checkpoints with EMA weights.

Usage:
    python scripts/train_full.py
    python scripts/train_full.py --resume checkpoints/latest.pt
    python scripts/train_full.py --batch_size 16 --total_steps 200000
"""

import argparse
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
from reswapper.models.ema import EMAModel
from reswapper.losses.reconstruction import ReconstructionLoss
from reswapper.losses.occlusion import OcclusionMaskLoss
from reswapper.data.dataset import VGGFace2Dataset
from reswapper.data.augmentations import FaceSwapAugmentation
from reswapper.training.scheduler import cosine_warmup_scheduler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--total_steps", type=int, default=200000)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--g_lr", type=float, default=1e-4)
    p.add_argument("--d_lr", type=float, default=4e-4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--d_reg_interval", type=int, default=16)
    p.add_argument("--wandb_project", type=str, default="reswapper-v2")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--dataset_dir", type=str, default="data/preprocessed_vggface2")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda")

    # ---- Models ----
    gen = FaceSwapGenerator(
        latent_channels=16, base_channels=256, bottleneck_channels=512,
    ).to(device)
    disc = ProjectedDiscriminator(backbone="dinov2_vitb14").to(device)
    vae = FluxVAE(pretrained_path="black-forest-labs/FLUX.1-schnell")
    vae.load(device, dtype=torch.bfloat16)
    ema = EMAModel(gen, decay=0.9999)

    # ---- Identity loss (PyTorch ArcFace on GPU) ----
    from iresnet import iresnet50
    arcface = iresnet50().to(device)
    arcface_path = "arcface_r50.pth"
    if os.path.exists(arcface_path):
        arcface.load_state_dict(torch.load(arcface_path, map_location=device))
    else:
        print("WARNING: arcface_r50.pth not found, identity loss disabled")
    arcface.eval()
    for p in arcface.parameters():
        p.requires_grad = False
    has_arcface = os.path.exists(arcface_path)

    # ---- LPIPS ----
    try:
        from reswapper.losses.perceptual import LPIPSLoss
        lpips_loss = LPIPSLoss()
        lpips_loss.load(device)
        has_lpips = True
        print("LPIPS loaded")
    except Exception as e:
        print(f"LPIPS unavailable: {e}")
        has_lpips = False

    # ---- Optimizers ----
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.g_lr, betas=(0.0, 0.99))
    opt_d = torch.optim.Adam(
        [p for p in disc.parameters() if p.requires_grad],
        lr=args.d_lr, betas=(0.0, 0.99),
    )
    sched_g = cosine_warmup_scheduler(opt_g, 5000, args.total_steps)
    sched_d = cosine_warmup_scheduler(opt_d, 5000, args.total_steps)

    occ_loss_fn = OcclusionMaskLoss()

    # ---- Data ----
    aug = FaceSwapAugmentation()
    ds = VGGFace2Dataset(
        dataset_dir=args.dataset_dir, resolution=1024,
        same_identity_ratio=0.5, transform=aug,
    )
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True,
    )

    # ---- Resume ----
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        gen.load_state_dict(ckpt["generator"])
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        if "optimizer_g" in ckpt:
            opt_g.load_state_dict(ckpt["optimizer_g"])
        if "optimizer_d" in ckpt:
            opt_d.load_state_dict(ckpt["optimizer_d"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from {args.resume} at step {start_step}")

    # ---- W&B ----
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, config=vars(args))
            print(f"W&B: {wandb_run.url}")
        except Exception as e:
            print(f"W&B init failed: {e}")

    os.makedirs("checkpoints", exist_ok=True)

    gen_params = sum(p.numel() for p in gen.parameters()) / 1e6
    disc_params = sum(p.numel() for p in disc.parameters() if p.requires_grad) / 1e6
    print(f"Training: {len(ds)} samples, batch={args.batch_size}, "
          f"steps={start_step}->{args.total_steps}")
    print(f"Gen: {gen_params:.1f}M, Disc: {disc_params:.1f}M trainable")
    print(f"Losses: adv=1.0, rec=10.0" +
          (", id=5.0" if has_arcface else "") +
          (", lpips=0.1" if has_lpips else ""))
    print()

    # ---- Training loop ----
    gen.train()
    disc.train()
    data_iter = iter(dl)
    t0 = time.time()

    for step in range(start_step, args.total_steps):
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
        B = target_image.shape[0]
        parsing = torch.zeros(B, 19, lh, lw, device=device)
        parsing[:, 1] = 1.0

        # ---- Generator forward ----
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            swapped_latent, swap_mask = gen(target_latent.float(), source_emb, parsing)
        output_latent = swap_mask * swapped_latent + (1 - swap_mask) * target_latent.float()
        output_pixels = vae.decode(output_latent)

        # ---- D step ----
        opt_d.zero_grad()
        real_logits = disc(target_image)
        fake_logits = disc(output_pixels.detach())
        d_loss = rpgan_loss_d(real_logits, fake_logits)
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in disc.parameters() if p.requires_grad], 1.0
        )
        opt_d.step()

        # R1+R2 penalty
        if step % args.d_reg_interval == 0:
            r1 = r1_penalty(disc, target_image.detach()) * 5.0
            r2 = r2_penalty(disc, output_pixels.detach()) * 5.0
            opt_d.zero_grad()
            (r1 + r2).backward()
            opt_d.step()

        # ---- G step ----
        opt_g.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            swapped_latent, swap_mask = gen(target_latent.float(), source_emb, parsing)
        output_latent = swap_mask * swapped_latent + (1 - swap_mask) * target_latent.float()
        output_pixels = vae.decode(output_latent)

        # Adversarial
        fake_logits_g = disc(output_pixels)
        real_logits_g = disc(target_image.detach())
        g_adv = rpgan_loss_g(real_logits_g, fake_logits_g)
        g_total = g_adv

        # Reconstruction (same-identity)
        g_rec = torch.tensor(0.0, device=device)
        if is_same.any():
            g_rec = F.l1_loss(output_pixels[is_same], target_image[is_same]) * 10.0
            g_total = g_total + g_rec

            # LPIPS on same-identity
            if has_lpips:
                g_lpips = lpips_loss(output_pixels[is_same], target_image[is_same]) * 0.1
                g_total = g_total + g_lpips

        # Identity loss (ArcFace cosine similarity)
        g_id = torch.tensor(0.0, device=device)
        if has_arcface:
            out_112 = F.interpolate(output_pixels, size=(112, 112), mode="bilinear", align_corners=False)
            out_emb = arcface(out_112 * 2.0 - 1.0)
            out_emb = F.normalize(out_emb, dim=1)
            src_emb_norm = F.normalize(source_emb, dim=1)
            g_id = (1.0 - F.cosine_similarity(out_emb, src_emb_norm, dim=1)).mean() * 5.0
            g_total = g_total + g_id

        # Mask losses
        mask_losses = occ_loss_fn(swap_mask)
        mask_total = torch.tensor(0.0, device=device)
        for v in mask_losses.values():
            g_total = g_total + v
            mask_total = mask_total + v

        g_total.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        opt_g.step()
        sched_g.step()
        sched_d.step()

        if step >= 5000:
            ema.update(gen)

        # ---- Logging ----
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            ms = elapsed / max(step - start_step + 1, 1) * 1000
            mem = torch.cuda.max_memory_allocated() / 1e9
            lr_g = opt_g.param_groups[0]["lr"]

            log_str = (
                f"[{step}/{args.total_steps}] "
                f"d={d_loss.item():.4f} g={g_adv.item():.4f} "
                f"rec={g_rec.item():.4f} id={g_id.item():.4f} "
                f"mask={swap_mask.mean().item():.3f} "
                f"lr={lr_g:.6f} mem={mem:.1f}GB {ms:.0f}ms/step"
            )
            print(log_str, flush=True)

            if wandb_run:
                import wandb
                wandb.log({
                    "loss/d_total": d_loss.item(),
                    "loss/g_adv": g_adv.item(),
                    "loss/g_rec": g_rec.item(),
                    "loss/g_id": g_id.item(),
                    "loss/g_total": g_total.item(),
                    "loss/mask": mask_total.item(),
                    "metrics/mask_mean": swap_mask.mean().item(),
                    "lr/generator": lr_g,
                    "timing/ms_per_step": ms,
                    "timing/gpu_gb": mem,
                }, step=step)

        # ---- Checkpoint ----
        if step > start_step and step % args.save_every == 0:
            path = f"checkpoints/checkpoint-{step}.pt"
            torch.save({
                "step": step,
                "generator": gen.state_dict(),
                "ema": ema.state_dict(),
                "optimizer_g": opt_g.state_dict(),
                "optimizer_d": opt_d.state_dict(),
            }, path)
            # Also save as latest for easy resume
            torch.save({
                "step": step,
                "generator": gen.state_dict(),
                "ema": ema.state_dict(),
                "optimizer_g": opt_g.state_dict(),
                "optimizer_d": opt_d.state_dict(),
            }, "checkpoints/latest.pt")
            print(f"Saved {path}", flush=True)

    # Final save
    torch.save(
        {"step": args.total_steps, "generator": gen.state_dict(), "ema": ema.state_dict()},
        f"checkpoints/checkpoint-{args.total_steps}.pt",
    )
    total_time = time.time() - t0
    print(f"\nTraining complete! {args.total_steps - start_step} steps in {total_time/3600:.1f}h")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
