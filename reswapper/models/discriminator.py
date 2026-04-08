import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectedDiscriminator(nn.Module):
    """Projected GAN discriminator with frozen DINOv2 backbone.

    Extracts features from multiple DINOv2 layers and applies lightweight
    trainable heads to produce real/fake predictions at each scale.

    No spectral norm or batch norm — stability from R3GAN's loss formulation
    (RpGAN + R1 + R2 gradient penalties) and fix-up initialization.

    References:
        - HP-GAN (arXiv:2602.03039) — FID 1.69 on FFHQ with frozen pretrained discriminator
        - R3GAN (arXiv:2501.05441) — principled GAN loss, no normalization needed
        - Projected GAN (arXiv:2111.01007) — frozen pretrained features for discrimination
    """

    def __init__(
        self,
        backbone: str = "dinov2_vitb14",
        extract_layers: list[int] = (2, 5, 8, 11),  # 0-indexed, DINOv2-B has blocks 0-11
        head_channels: int = 256,
    ):
        super().__init__()
        self.extract_layers = list(extract_layers)

        # Load frozen DINOv2 backbone
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # DINOv2-B/14: patch_size=14, embed_dim=768
        self.patch_size = self.backbone.patch_size
        feat_dim = self.backbone.embed_dim

        # Trainable discriminator heads — one per extracted layer
        self.heads = nn.ModuleList()
        for _ in extract_layers:
            head = nn.Sequential(
                nn.Conv2d(feat_dim, head_channels, 3, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(head_channels, 1, 3, padding=1, bias=False),
            )
            # Fix-up init: zero-init last conv
            nn.init.kaiming_normal_(head[0].weight)
            nn.init.zeros_(head[2].weight)
            self.heads.append(head)

    def _extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract intermediate features from DINOv2 at specified layers.

        Args:
            x: pixel-space images [B, 3, H, W] in [0, 1] range
        Returns:
            list of feature maps [B, embed_dim, h, w] at each extracted layer
        """
        # DINOv2 expects images normalized with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # Resize to be divisible by patch_size
        B, C, H, W = x.shape
        h = H // self.patch_size
        w = W // self.patch_size
        x = F.interpolate(x, size=(h * self.patch_size, w * self.patch_size), mode="bilinear", align_corners=False)

        # Get intermediate features using DINOv2's forward
        features = self.backbone.get_intermediate_layers(x, n=self.extract_layers, reshape=True)
        return list(features)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: pixel-space images [B, 3, H, W] in [0, 1] range
        Returns:
            list of per-patch logits from each head
        """
        # Enable gradient flow through the backbone for R1/R2 penalties
        # (backbone weights are frozen, but gradients w.r.t. input must flow)
        features = self._extract_features(x)

        logits = []
        for feat, head in zip(features, self.heads):
            logits.append(head(feat))
        return logits


def rpgan_loss_d(real_logits: list[torch.Tensor], fake_logits: list[torch.Tensor]) -> torch.Tensor:
    """Regularized relativistic paired GAN loss for discriminator.

    L_D = -E[log sigmoid(D(real) - D(fake))]
    """
    loss = 0.0
    for r, f in zip(real_logits, fake_logits):
        loss = loss + F.softplus(-(r - f)).mean()
    return loss / len(real_logits)


def rpgan_loss_g(real_logits: list[torch.Tensor], fake_logits: list[torch.Tensor]) -> torch.Tensor:
    """Regularized relativistic paired GAN loss for generator.

    L_G = -E[log sigmoid(D(fake) - D(real))]
    """
    loss = 0.0
    for r, f in zip(real_logits, fake_logits):
        loss = loss + F.softplus(-(f - r)).mean()
    return loss / len(real_logits)


def r1_penalty(discriminator: nn.Module, real_images: torch.Tensor, max_size: int = 256) -> torch.Tensor:
    """R1 gradient penalty on real data.

    Downscales images before computing penalty to save memory — the MATH
    attention backend needed for second-order gradients uses much more
    memory than efficient attention.

    Args:
        discriminator: the discriminator module
        real_images: [B, 3, H, W] real images
        max_size: downscale to this resolution for penalty computation
    """
    # Downscale to fit in memory with MATH attention
    if real_images.shape[-1] > max_size:
        real_images = F.interpolate(real_images, size=(max_size, max_size), mode="bilinear", align_corners=False)

    real_images = real_images.detach().requires_grad_(True)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        real_logits = discriminator(real_images)
    total = sum(l.sum() for l in real_logits)
    grads = torch.autograd.grad(outputs=total, inputs=real_images, create_graph=True)[0]
    return grads.pow(2).reshape(grads.shape[0], -1).sum(1).mean()


def r2_penalty(discriminator: nn.Module, fake_images: torch.Tensor, max_size: int = 256) -> torch.Tensor:
    """R2 gradient penalty on fake data."""
    if fake_images.shape[-1] > max_size:
        fake_images = F.interpolate(fake_images, size=(max_size, max_size), mode="bilinear", align_corners=False)

    fake_images = fake_images.detach().requires_grad_(True)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        fake_logits = discriminator(fake_images)
    total = sum(l.sum() for l in fake_logits)
    grads = torch.autograd.grad(outputs=total, inputs=fake_images, create_graph=True)[0]
    return grads.pow(2).reshape(grads.shape[0], -1).sum(1).mean()
