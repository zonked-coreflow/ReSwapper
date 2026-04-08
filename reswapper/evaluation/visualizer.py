"""Validation visualization for face swapping training.

Generates grid images showing:
source | target | swapped | swap_mask | difference
"""

import torch
import torchvision.utils as vutils


def make_validation_grid(
    source_images: torch.Tensor,
    target_images: torch.Tensor,
    swapped_images: torch.Tensor,
    swap_masks: torch.Tensor,
    max_samples: int = 8,
) -> torch.Tensor:
    """Create a validation grid image.

    Args:
        source_images: [B, 3, H, W] source identity faces in [0, 1]
        target_images: [B, 3, H, W] target faces in [0, 1]
        swapped_images: [B, 3, H, W] generated swapped faces in [0, 1]
        swap_masks: [B, 1, H, W] swap masks in [0, 1]
        max_samples: max number of rows to show
    Returns:
        grid: [3, grid_H, grid_W] tensor ready for logging
    """
    n = min(max_samples, source_images.shape[0])

    source = source_images[:n].cpu()
    target = target_images[:n].cpu()
    swapped = swapped_images[:n].cpu().clamp(0, 1)

    # Expand mask to 3 channels for visualization
    masks = swap_masks[:n].cpu()
    if masks.shape[-2:] != target.shape[-2:]:
        masks = torch.nn.functional.interpolate(
            masks, size=target.shape[-2:], mode="bilinear", align_corners=False
        )
    masks_3ch = masks.expand(-1, 3, -1, -1)

    # Difference map (abs diff between swapped and target)
    diff = (swapped - target).abs().clamp(0, 1)

    # Interleave: source, target, swapped, mask, diff for each sample
    rows = []
    for i in range(n):
        rows.extend([source[i], target[i], swapped[i], masks_3ch[i], diff[i]])

    grid = vutils.make_grid(rows, nrow=5, padding=2, pad_value=0.5)
    return grid


def save_validation_grid(
    grid: torch.Tensor,
    path: str,
):
    """Save a validation grid to disk."""
    from torchvision.utils import save_image
    save_image(grid, path)


def log_validation_to_wandb(
    grid: torch.Tensor,
    step: int,
    caption: str = "validation",
):
    """Log a validation grid to Weights & Biases."""
    try:
        import wandb
        # Convert to [H, W, 3] numpy for wandb
        img = grid.permute(1, 2, 0).numpy()
        img = (img * 255).clip(0, 255).astype("uint8")
        wandb.log({caption: wandb.Image(img)}, step=step)
    except ImportError:
        pass
