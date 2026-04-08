from dataclasses import dataclass, field
from typing import List, Optional, Literal
from pathlib import Path
import yaml


@dataclass
class DataConfig:
    dataset_dir: str = "data/preprocessed_vggface2"
    ffhq_dir: Optional[str] = None
    num_workers: int = 16
    pin_memory: bool = True
    resolution: int = 512
    same_identity_ratio: float = 0.5
    occlusion_augment_ratio: float = 0.3


@dataclass
class AugmentationConfig:
    horizontal_flip: float = 0.5
    color_jitter: bool = True
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1
    random_grayscale: float = 0.1
    jpeg_quality_range: List[int] = field(default_factory=lambda: [60, 100])
    jpeg_probability: float = 0.3
    gaussian_blur_probability: float = 0.1
    gaussian_noise_std: float = 0.02
    gaussian_noise_probability: float = 0.2
    random_erasing: float = 0.05


@dataclass
class GeneratorConfig:
    base_channels: int = 256
    bottleneck_channels: int = 512
    latent_channels: int = 16  # FLUX VAE latent channels
    id_dim: int = 512
    num_id_tokens: int = 16
    cross_attn_resolutions: List[int] = field(default_factory=lambda: [8, 16, 32])
    adain_resolutions: List[int] = field(default_factory=lambda: [64])
    num_bottleneck_blocks: int = 2
    group_size: int = 16  # R3GAN grouped conv
    expansion: float = 1.5  # R3GAN inverted bottleneck
    gradient_checkpointing: bool = False
    parsing_channels: int = 19  # SegFace classes


@dataclass
class DiscriminatorConfig:
    backbone: str = "dinov2_vitb14"
    extract_layers: List[int] = field(default_factory=lambda: [2, 5, 8, 11])  # 0-indexed, DINOv2-B has blocks 0-11
    head_channels: int = 256


@dataclass
class IdentityConfig:
    model: str = "blendface"  # "blendface", "arcface", "topofr"
    weights_path: Optional[str] = None
    embedding_dim: int = 512
    num_tokens: int = 16
    freeze_backbone: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 32
    total_steps: int = 500_000
    g_lr: float = 1e-4
    d_lr: float = 4e-4
    g_betas: List[float] = field(default_factory=lambda: [0.0, 0.99])
    d_betas: List[float] = field(default_factory=lambda: [0.0, 0.99])
    warmup_steps: int = 5000
    lr_schedule: str = "cosine"
    min_lr_ratio: float = 0.01
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    gradient_accumulation_steps: int = 1
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.9999
    ema_start_step: int = 5000
    d_reg_interval: int = 16
    r1_gamma: float = 10.0
    r2_gamma: float = 10.0


@dataclass
class LossConfig:
    # Reconstruction (same-identity only)
    l1_weight: float = 10.0
    lpips_weight: float = 0.1
    pdino_weight: float = 0.01
    # Identity (always)
    identity_weight: float = 5.0
    # Adversarial (always)
    adversarial_weight: float = 1.0
    # Geometric preservation (always)
    gaze_weight: float = 0.5
    expression_weight: float = 0.3
    landmark_weight: float = 0.1
    # Occlusion mask
    mask_bce_weight: float = 5.0
    mask_reg_weight: float = 0.5
    mask_tv_weight: float = 0.1


@dataclass
class CheckpointConfig:
    save_dir: str = "checkpoints"
    save_every_steps: int = 5000
    keep_top_k: int = 5
    metric_for_best: str = "val/identity_similarity"
    metric_mode: str = "max"
    resume_from: Optional[str] = None


@dataclass
class LoggingConfig:
    log_every_steps: int = 100
    val_every_steps: int = 5000
    val_num_samples: int = 64
    use_wandb: bool = True
    wandb_project: str = "reswapper-v2"
    wandb_entity: Optional[str] = None
    num_preview_samples: int = 8


@dataclass
class ReSwapperConfig:
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "ReSwapperConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        config = cls()
        _merge_dict_into_dataclass(config, raw)
        return config

    @classmethod
    def from_yamls(cls, paths: List[str]) -> "ReSwapperConfig":
        config = cls()
        for path in paths:
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            _merge_dict_into_dataclass(config, raw)
        return config


def _merge_dict_into_dataclass(dc, d: dict):
    for key, value in d.items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if hasattr(current, '__dataclass_fields__') and isinstance(value, dict):
            _merge_dict_into_dataclass(current, value)
        else:
            setattr(dc, key, value)
