import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """L1 reconstruction loss for same-identity training pairs.

    When source and target are the same identity, the swapped output
    should reconstruct the target face. This provides direct pixel-level
    supervision without needing a teacher model (InSwapper).
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 3, H, W] in [0, 1]
            target: [B, 3, H, W] in [0, 1]
        Returns:
            scalar L1 loss
        """
        return F.l1_loss(pred, target)
