import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from reswapper.config import ReSwapperConfig
from reswapper.models.generator import FaceSwapGenerator
from reswapper.models.discriminator import (
    ProjectedDiscriminator,
    rpgan_loss_d,
    rpgan_loss_g,
    r1_penalty,
    r2_penalty,
)
from reswapper.models.vae import FluxVAE
from reswapper.models.identity_encoder import IdentityEncoder
from reswapper.models.face_parser import FaceParser
from reswapper.models.ema import EMAModel
from reswapper.losses.identity import IdentityLoss
from reswapper.losses.perceptual import LPIPSLoss, PDINOLoss
from reswapper.losses.reconstruction import ReconstructionLoss
from reswapper.losses.occlusion import OcclusionMaskLoss
from reswapper.losses.geometric import LandmarkLoss, GazeLoss, ExpressionLoss
from reswapper.training.scheduler import cosine_warmup_scheduler
from reswapper.training.checkpoint import CheckpointManager
from reswapper.data.dataset import VGGFace2Dataset
from reswapper.data.augmentations import FaceSwapAugmentation
from reswapper.data.occlusion_synth import SyntheticOcclusionAugmentation
from reswapper.utils.distributed import is_main_process


class Trainer:
    """Main training loop for ReSwapper v2.

    Supports:
    - bf16 mixed precision (B200 native)
    - R3GAN G/D alternation with gradient accumulation
    - Gradient checkpointing
    - EMA on generator weights
    - Cosine warmup LR schedule
    - Top-K checkpoint manager
    - W&B logging
    - DDP for multi-GPU
    """

    def __init__(self, config: ReSwapperConfig, local_rank: int = 0, world_size: int = 1):
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        self._build_models()
        self._build_losses()
        self._build_optimizers()
        self._build_data()
        self._build_logging()

        self.global_step = 0
        self.occlusion_augmenter = SyntheticOcclusionAugmentation(
            probability=config.data.occlusion_augment_ratio
        )

        if config.checkpoint.resume_from:
            self._resume(config.checkpoint.resume_from)

    def _build_models(self):
        cfg = self.config

        # Generator
        self.generator = FaceSwapGenerator(
            latent_channels=cfg.generator.latent_channels,
            base_channels=cfg.generator.base_channels,
            bottleneck_channels=cfg.generator.bottleneck_channels,
            id_dim=cfg.identity.embedding_dim,
            num_id_tokens=cfg.identity.num_tokens,
            parsing_channels=cfg.generator.parsing_channels,
            group_size=cfg.generator.group_size,
            expansion=cfg.generator.expansion,
            num_bottleneck_blocks=cfg.generator.num_bottleneck_blocks,
            gradient_checkpointing=cfg.generator.gradient_checkpointing,
        ).to(self.device)

        # Discriminator
        self.discriminator = ProjectedDiscriminator(
            backbone=cfg.discriminator.backbone,
            extract_layers=cfg.discriminator.extract_layers,
            head_channels=cfg.discriminator.head_channels,
        ).to(self.device)

        # VAE (frozen)
        self.vae = FluxVAE()
        self.vae.load(self.device, dtype=torch.bfloat16 if cfg.training.amp_dtype == "bfloat16" else torch.float32)

        # Identity encoder (frozen)
        self.identity_encoder = IdentityEncoder(
            model=cfg.identity.model,
            weights_path=cfg.identity.weights_path,
        )
        self.identity_encoder.load(self.device)

        # Face parser (frozen)
        self.face_parser = FaceParser(num_classes=cfg.generator.parsing_channels)
        self.face_parser.load(self.device)

        # EMA
        self.ema = EMAModel(self.generator, decay=cfg.training.ema_decay)
        self.ema.to(self.device)

        # DDP wrapping
        if self.world_size > 1:
            self.generator = nn.parallel.DistributedDataParallel(
                self.generator, device_ids=[self.local_rank]
            )
            self.discriminator = nn.parallel.DistributedDataParallel(
                self.discriminator, device_ids=[self.local_rank]
            )

    def _build_losses(self):
        cfg = self.config.loss

        self.identity_loss = IdentityLoss(self.identity_encoder)
        self.reconstruction_loss = ReconstructionLoss()
        self.occlusion_loss = OcclusionMaskLoss(
            bce_weight=cfg.mask_bce_weight,
            reg_weight=cfg.mask_reg_weight,
            tv_weight=cfg.mask_tv_weight,
        )

        # LPIPS (needs separate loading)
        self.lpips_loss = LPIPSLoss()
        try:
            self.lpips_loss.load(self.device)
        except Exception as e:
            print(f"Warning: Failed to load LPIPS: {e}")
            self.lpips_loss = None

        # P-DINO (shares DINOv2 with discriminator if possible)
        self.pdino_loss = PDINOLoss()
        try:
            self.pdino_loss.load(self.device)
        except Exception as e:
            print(f"Warning: Failed to load P-DINO: {e}")
            self.pdino_loss = None

        # Geometric losses (placeholder loading)
        self.landmark_loss = LandmarkLoss()
        self.landmark_loss.load(self.device)
        self.gaze_loss = GazeLoss()
        self.gaze_loss.load(self.device)
        self.expression_loss = ExpressionLoss()
        self.expression_loss.load(self.device)

    def _build_optimizers(self):
        cfg = self.config.training

        # Generator optimizer
        gen_module = self.generator.module if hasattr(self.generator, 'module') else self.generator
        self.optimizer_g = torch.optim.Adam(
            gen_module.parameters(),
            lr=cfg.g_lr,
            betas=tuple(cfg.g_betas),
        )

        # Discriminator optimizer (only trainable heads)
        disc_module = self.discriminator.module if hasattr(self.discriminator, 'module') else self.discriminator
        self.optimizer_d = torch.optim.Adam(
            [p for p in disc_module.parameters() if p.requires_grad],
            lr=cfg.d_lr,
            betas=tuple(cfg.d_betas),
        )

        self.scheduler_g = cosine_warmup_scheduler(
            self.optimizer_g, cfg.warmup_steps, cfg.total_steps, cfg.min_lr_ratio
        )
        self.scheduler_d = cosine_warmup_scheduler(
            self.optimizer_d, cfg.warmup_steps, cfg.total_steps, cfg.min_lr_ratio
        )

        # AMP scaler
        amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
        self.amp_dtype = amp_dtype
        # bf16 doesn't need GradScaler
        self.use_scaler = cfg.use_amp and amp_dtype == torch.float16
        self.scaler_g = torch.amp.GradScaler(enabled=self.use_scaler)
        self.scaler_d = torch.amp.GradScaler(enabled=self.use_scaler)

    def _build_data(self):
        cfg = self.config

        augmentation = FaceSwapAugmentation(
            horizontal_flip=cfg.augmentation.horizontal_flip,
            color_jitter=cfg.augmentation.color_jitter,
            brightness=cfg.augmentation.brightness,
            contrast=cfg.augmentation.contrast,
            saturation=cfg.augmentation.saturation,
            hue=cfg.augmentation.hue,
            random_grayscale=cfg.augmentation.random_grayscale,
            jpeg_quality_range=cfg.augmentation.jpeg_quality_range,
            jpeg_probability=cfg.augmentation.jpeg_probability,
            gaussian_blur_probability=cfg.augmentation.gaussian_blur_probability,
            gaussian_noise_std=cfg.augmentation.gaussian_noise_std,
            gaussian_noise_probability=cfg.augmentation.gaussian_noise_probability,
            random_erasing=cfg.augmentation.random_erasing,
        )

        self.dataset = VGGFace2Dataset(
            dataset_dir=cfg.data.dataset_dir,
            same_identity_ratio=cfg.data.same_identity_ratio,
            resolution=cfg.data.resolution,
            transform=augmentation,
        )

        sampler = DistributedSampler(self.dataset) if self.world_size > 1 else None

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=cfg.training.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            drop_last=True,
        )

    def _build_logging(self):
        self.wandb_run = None
        if self.config.logging.use_wandb and is_main_process():
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.config.logging.wandb_project,
                    entity=self.config.logging.wandb_entity,
                    config=vars(self.config),
                )
            except Exception as e:
                print(f"Warning: Failed to init wandb: {e}")

        self.checkpoint_manager = CheckpointManager(
            save_dir=self.config.checkpoint.save_dir,
            keep_top_k=self.config.checkpoint.keep_top_k,
            metric_mode=self.config.checkpoint.metric_mode,
        )

    def train(self):
        """Main training loop."""
        cfg = self.config.training
        self.generator.train()
        self.discriminator.train()

        data_iter = iter(self.dataloader)

        for step in range(self.global_step, cfg.total_steps):
            t0 = time.time()

            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Training step
            metrics = self._train_step(batch, step)

            # EMA update
            gen_module = self.generator.module if hasattr(self.generator, 'module') else self.generator
            if step >= cfg.ema_start_step:
                self.ema.update(gen_module)

            self.global_step = step + 1
            elapsed = time.time() - t0

            # Logging
            if step % self.config.logging.log_every_steps == 0 and is_main_process():
                metrics["timing/step_ms"] = elapsed * 1000
                metrics["lr/generator"] = self.optimizer_g.param_groups[0]["lr"]
                metrics["lr/discriminator"] = self.optimizer_d.param_groups[0]["lr"]
                self._log_metrics(metrics, step)

            # Validation + Checkpoint
            if step % self.config.logging.val_every_steps == 0 and step > 0 and is_main_process():
                self.checkpoint_manager.save(self._get_state(), step)
                print(f"Step {step}: checkpoint saved")

            if step % self.config.checkpoint.save_every_steps == 0 and step > 0 and is_main_process():
                self.checkpoint_manager.save(self._get_state(), step)

    def _train_step(self, batch: dict, step: int) -> dict:
        """Single training step with G and D updates."""
        cfg_t = self.config.training
        cfg_l = self.config.loss
        metrics = {}

        target_image = batch["target_image"]  # [B, 3, H, W]
        source_embedding = batch["source_embedding"]  # [B, 512]
        is_same_identity = batch["is_same_identity"]  # [B]

        # Get or compute target latent and parsing
        if "target_latent" in batch:
            target_latent = batch["target_latent"]
        else:
            target_latent = self.vae.encode(target_image)

        if "target_parsing" in batch:
            target_parsing = batch["target_parsing"]
        else:
            latent_h, latent_w = target_latent.shape[2], target_latent.shape[3]
            target_parsing = self.face_parser(target_image, target_size=(latent_h, latent_w))

        # Resize parsing to latent spatial size if needed
        if target_parsing.shape[-1] != target_latent.shape[-1]:
            target_parsing = F.interpolate(
                target_parsing,
                size=(target_latent.shape[2], target_latent.shape[3]),
                mode="bilinear",
                align_corners=False,
            )

        # === Generator forward ===
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.config.training.use_amp):
            swapped_latent, swap_mask = self.generator(target_latent, source_embedding, target_parsing)

            # Composite in latent space
            output_latent = swap_mask * swapped_latent + (1 - swap_mask) * target_latent

        # Decode to pixel space for losses
        output_pixels = self.vae.decode(output_latent.float())
        target_pixels = target_image

        # === Discriminator step ===
        self.optimizer_d.zero_grad()
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=cfg_t.use_amp):
            real_logits = self.discriminator(target_pixels.detach())
            fake_logits = self.discriminator(output_pixels.detach())
            d_loss = rpgan_loss_d(real_logits, fake_logits)

        if self.use_scaler:
            self.scaler_d.scale(d_loss).backward()
            self.scaler_d.unscale_(self.optimizer_d)
        else:
            d_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.discriminator.parameters() if p.requires_grad],
            cfg_t.grad_clip_norm,
        )
        if self.use_scaler:
            self.scaler_d.step(self.optimizer_d)
            self.scaler_d.update()
        else:
            self.optimizer_d.step()

        # R1 + R2 gradient penalties
        if step % cfg_t.d_reg_interval == 0:
            r1 = r1_penalty(self.discriminator, target_pixels.detach()) * cfg_t.r1_gamma / 2
            r2 = r2_penalty(self.discriminator, output_pixels.detach()) * cfg_t.r2_gamma / 2
            reg_loss = r1 + r2
            self.optimizer_d.zero_grad()
            reg_loss.backward()
            self.optimizer_d.step()
            metrics["loss/r1"] = r1.item()
            metrics["loss/r2"] = r2.item()

        self.scheduler_d.step()
        metrics["loss/d_total"] = d_loss.item()

        # === Generator step ===
        self.optimizer_g.zero_grad()
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=cfg_t.use_amp):
            # Re-forward for generator gradients
            swapped_latent, swap_mask = self.generator(target_latent, source_embedding, target_parsing)
            output_latent = swap_mask * swapped_latent + (1 - swap_mask) * target_latent

        output_pixels = self.vae.decode(output_latent.float())

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=cfg_t.use_amp):
            # Adversarial loss
            fake_logits_g = self.discriminator(output_pixels)
            real_logits_g = self.discriminator(target_pixels.detach())
            g_adv = rpgan_loss_g(real_logits_g, fake_logits_g) * cfg_l.adversarial_weight

            # Identity loss
            g_id = self.identity_loss(output_pixels, source_embedding) * cfg_l.identity_weight

            g_total = g_adv + g_id

            # Reconstruction losses (same-identity only)
            if is_same_identity.any():
                same = is_same_identity
                if same.sum() > 0:
                    g_rec = self.reconstruction_loss(output_pixels[same], target_pixels[same]) * cfg_l.l1_weight
                    g_total = g_total + g_rec
                    metrics["loss/g_reconstruction"] = g_rec.item()

                    if self.lpips_loss is not None:
                        g_lpips = self.lpips_loss(output_pixels[same], target_pixels[same]) * cfg_l.lpips_weight
                        g_total = g_total + g_lpips
                        metrics["loss/g_lpips"] = g_lpips.item()

            # Geometric losses
            g_gaze = self.gaze_loss(output_pixels, target_pixels) * cfg_l.gaze_weight
            g_expr = self.expression_loss(output_pixels, target_pixels) * cfg_l.expression_weight
            g_lm = self.landmark_loss(output_pixels, target_pixels) * cfg_l.landmark_weight
            g_total = g_total + g_gaze + g_expr + g_lm

            # Occlusion mask losses
            mask_losses = self.occlusion_loss(swap_mask)
            for k, v in mask_losses.items():
                g_total = g_total + v
                metrics[f"loss/{k}"] = v.item()

        if self.use_scaler:
            self.scaler_g.scale(g_total).backward()
            self.scaler_g.unscale_(self.optimizer_g)
        else:
            g_total.backward()

        gen_params = self.generator.module.parameters() if hasattr(self.generator, 'module') else self.generator.parameters()
        torch.nn.utils.clip_grad_norm_(gen_params, cfg_t.grad_clip_norm)

        if self.use_scaler:
            self.scaler_g.step(self.optimizer_g)
            self.scaler_g.update()
        else:
            self.optimizer_g.step()

        self.scheduler_g.step()

        metrics["loss/g_total"] = g_total.item()
        metrics["loss/g_adversarial"] = g_adv.item()
        metrics["loss/g_identity"] = g_id.item()

        return metrics

    def _log_metrics(self, metrics: dict, step: int):
        parts = [f"Step {step}"]
        for k in ["loss/g_total", "loss/d_total", "loss/g_identity", "loss/g_reconstruction"]:
            if k in metrics:
                parts.append(f"{k.split('/')[-1]}={metrics[k]:.4f}")
        if "timing/step_ms" in metrics:
            parts.append(f"{metrics['timing/step_ms']:.0f}ms")
        print(" | ".join(parts))

        if self.wandb_run:
            import wandb
            wandb.log(metrics, step=step)

    def _get_state(self) -> dict:
        gen_module = self.generator.module if hasattr(self.generator, 'module') else self.generator
        disc_module = self.discriminator.module if hasattr(self.discriminator, 'module') else self.discriminator
        return {
            "step": self.global_step,
            "generator": gen_module.state_dict(),
            "discriminator": {k: v for k, v in disc_module.state_dict().items()
                              if "backbone" not in k},  # Don't save frozen DINOv2
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "scheduler_g": self.scheduler_g.state_dict(),
            "scheduler_d": self.scheduler_d.state_dict(),
            "ema": self.ema.state_dict(),
        }

    def _resume(self, path: str):
        state = self.checkpoint_manager.load(path)
        gen_module = self.generator.module if hasattr(self.generator, 'module') else self.generator
        gen_module.load_state_dict(state["generator"])
        self.optimizer_g.load_state_dict(state["optimizer_g"])
        self.optimizer_d.load_state_dict(state["optimizer_d"])
        self.scheduler_g.load_state_dict(state["scheduler_g"])
        self.scheduler_d.load_state_dict(state["scheduler_d"])
        self.ema.load_state_dict(state["ema"])
        self.global_step = state["step"]
        print(f"Resumed from {path} at step {self.global_step}")
