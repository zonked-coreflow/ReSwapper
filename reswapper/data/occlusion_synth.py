import random

import torch
import torch.nn.functional as F


class SyntheticOcclusionAugmentation:
    """Synthetic occlusion compositing for training the swap mask.

    Overlays random shapes, textures, and patches onto face images with
    known alpha masks. The generator learns to predict these occluded regions
    via the mask BCE loss.

    Reference:
        - CelebAMat (FaceMat, arXiv:2508.03055) — compositing occlusions
          from multiple sources with random size/orientation
        - VividFace (arXiv:2412.11279) — comprehensive occlusion augmentation
    """

    def __init__(
        self,
        probability: float = 0.3,
        num_occlusions_range: tuple[int, int] = (1, 3),
        size_range: tuple[float, float] = (0.05, 0.25),
    ):
        self.probability = probability
        self.num_occlusions_range = num_occlusions_range
        self.size_range = size_range

    def __call__(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Args:
            image: [3, H, W] face image in [0, 1]
        Returns:
            augmented_image: [3, H, W] with occlusions
            occlusion_mask: [1, H, W] where 1=face (swap), 0=occluded (keep target)
            is_augmented: bool
        """
        H, W = image.shape[1], image.shape[2]
        mask = torch.ones(1, H, W, device=image.device)

        if random.random() > self.probability:
            return image, mask, False

        num_occlusions = random.randint(*self.num_occlusions_range)
        augmented = image.clone()

        for _ in range(num_occlusions):
            occlusion_type = random.choice(["rectangle", "ellipse", "noise_patch"])

            # Random size and position
            size_frac = random.uniform(*self.size_range)
            oh = int(H * size_frac)
            ow = int(W * size_frac * random.uniform(0.5, 2.0))
            oh = min(oh, H - 1)
            ow = min(ow, W - 1)
            top = random.randint(0, H - oh)
            left = random.randint(0, W - ow)

            if occlusion_type == "rectangle":
                # Solid color rectangle
                color = torch.rand(3, 1, 1, device=image.device)
                augmented[:, top:top + oh, left:left + ow] = color
                mask[:, top:top + oh, left:left + ow] = 0.0

            elif occlusion_type == "ellipse":
                # Elliptical occlusion with soft edges
                yy, xx = torch.meshgrid(
                    torch.linspace(-1, 1, oh, device=image.device),
                    torch.linspace(-1, 1, ow, device=image.device),
                    indexing="ij",
                )
                ellipse = (xx ** 2 + yy ** 2) < 1.0
                alpha = ellipse.float().unsqueeze(0)
                # Gaussian blur the edges
                if oh > 4 and ow > 4:
                    alpha = _gaussian_blur_2d(alpha, kernel_size=5, sigma=1.5)

                color = torch.rand(3, 1, 1, device=image.device).expand(3, oh, ow)
                augmented[:, top:top + oh, left:left + ow] = (
                    augmented[:, top:top + oh, left:left + ow] * (1 - alpha) + color * alpha
                )
                mask[:, top:top + oh, left:left + ow] = (
                    mask[:, top:top + oh, left:left + ow] * (1 - alpha)
                )

            elif occlusion_type == "noise_patch":
                # Textured noise patch
                noise = torch.rand(3, oh, ow, device=image.device)
                augmented[:, top:top + oh, left:left + ow] = noise
                mask[:, top:top + oh, left:left + ow] = 0.0

        return augmented.clamp(0, 1), mask, True


def _gaussian_blur_2d(
    x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.5
) -> torch.Tensor:
    """Apply 2D Gaussian blur to a [1, H, W] tensor."""
    k = kernel_size
    ax = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
    kernel = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel_2d = kernel.unsqueeze(0) * kernel.unsqueeze(1)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]

    padding = k // 2
    return F.conv2d(x.unsqueeze(0), kernel_2d, padding=padding).squeeze(0)
