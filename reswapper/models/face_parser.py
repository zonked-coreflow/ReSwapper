import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceParser(nn.Module):
    """Face parsing model wrapper for generating semantic segmentation maps.

    Produces 19-class parsing maps used as additional generator input
    for occlusion awareness. The parser is frozen during training.

    Supports SegFace (recommended, 88.96 F1) or BiSeNet (fallback).

    19 CelebAMask-HQ classes:
        0: background, 1: skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye,
        6: eye_g (glasses), 7: l_ear, 8: r_ear, 9: ear_r (earring),
        10: nose, 11: mouth, 12: u_lip, 13: l_lip, 14: neck,
        15: necklace, 16: cloth, 17: hair, 18: hat
    """

    def __init__(self, num_classes: int = 19):
        super().__init__()
        self.num_classes = num_classes
        self.model = None

    def load(self, device: torch.device = torch.device("cpu"), weights_path: str | None = None):
        """Load the face parsing model."""
        if weights_path:
            self.model = torch.load(weights_path, map_location=device)
        else:
            # Lightweight placeholder: use a small segmentation network
            # Replace with SegFace or BiSeNet when weights are available
            self.model = _PlaceholderParser(self.num_classes).to(device)
            print("Warning: Using placeholder face parser. Download SegFace weights "
                  "for production use.")

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor, target_size: tuple[int, int] | None = None) -> torch.Tensor:
        """Generate face parsing maps.

        Args:
            images: [B, 3, H, W] in [0, 1] range
            target_size: optional (h, w) to resize parsing map
        Returns:
            parsing: [B, num_classes, h, w] one-hot encoded parsing map
        """
        if self.model is None:
            B, _, H, W = images.shape
            h, w = target_size if target_size else (H, W)
            # Return uniform parsing (all skin) as fallback
            parsing = torch.zeros(B, self.num_classes, h, w, device=images.device)
            parsing[:, 1] = 1.0  # skin
            return parsing

        logits = self.model(images)

        if target_size is not None:
            logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        # Convert to one-hot via argmax + scatter
        classes = logits.argmax(dim=1)  # [B, H, W]
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, classes.unsqueeze(1), 1.0)
        return one_hot


class _PlaceholderParser(nn.Module):
    """Lightweight placeholder segmentation network for testing."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, x):
        return self.net(x)
