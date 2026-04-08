import torch
import torch.nn as nn
import torch.nn.functional as F


class LPIPSLoss(nn.Module):
    """LPIPS perceptual loss wrapper.

    Uses VGG-based learned perceptual similarity.
    Only applied during reconstruction (same-identity) training.
    """

    def __init__(self):
        super().__init__()
        self.lpips = None

    def load(self, device: torch.device = torch.device("cpu")):
        import lpips
        self.lpips = lpips.LPIPS(net="vgg").to(device)
        self.lpips.eval()
        for p in self.lpips.parameters():
            p.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 3, H, W] in [0, 1]
            target: [B, 3, H, W] in [0, 1]
        Returns:
            scalar loss
        """
        # LPIPS expects [-1, 1]
        return self.lpips(pred * 2 - 1, target * 2 - 1).mean()


class PDINOLoss(nn.Module):
    """P-DINO perceptual loss using frozen DINOv2-B patch features.

    Computes cosine distance between DINOv2 patch-level features of
    predicted and target images. Captures global semantic structure
    better than LPIPS which focuses on local textures.

    Reference: PixelGen (arXiv:2602.02493) — FID improvement 10.0 → 7.46
    when combined with LPIPS.
    """

    def __init__(self):
        super().__init__()
        self.backbone = None

    def load(self, device: torch.device = torch.device("cpu")):
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.backbone.eval()
        self.backbone = self.backbone.to(device)
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract patch features from final layer of DINOv2-B."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        x = (images - mean) / std

        # Resize to be divisible by patch_size=14
        ps = 14
        H, W = x.shape[-2:]
        h, w = H // ps * ps, W // ps * ps
        if h != H or w != W:
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

        # Get final layer features
        features = self.backbone.get_intermediate_layers(x, n=[12], reshape=False)
        return features[0]  # [B, num_patches, 768]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 3, H, W] in [0, 1]
            target: [B, 3, H, W] in [0, 1]
        Returns:
            scalar loss: mean(1 - cosine_similarity) over patches
        """
        with torch.no_grad():
            target_features = self._extract_patch_features(target)

        pred_features = self._extract_patch_features(pred)

        similarity = F.cosine_similarity(pred_features, target_features, dim=-1)
        loss = (1.0 - similarity).mean()
        return loss
