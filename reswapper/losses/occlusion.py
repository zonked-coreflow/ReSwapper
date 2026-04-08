import torch
import torch.nn as nn
import torch.nn.functional as F


class OcclusionMaskLoss(nn.Module):
    """Losses for training the occlusion-aware swap mask.

    The generator predicts a per-pixel swap mask (1=swap, 0=keep target).
    This module provides three losses:
    1. BCE loss on augmented samples with known GT occlusion masks
    2. Regularization toward 1.0 on clean (non-augmented) samples
    3. Total variation smoothness on the mask

    References:
        - SelfSwapper (arXiv:2402.07370) — M = M_ras - M_occ
        - FaceMat (arXiv:2508.03055) — alpha matting for soft boundaries
    """

    def __init__(
        self,
        bce_weight: float = 5.0,
        reg_weight: float = 0.5,
        tv_weight: float = 0.1,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.reg_weight = reg_weight
        self.tv_weight = tv_weight

    def forward(
        self,
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor | None = None,
        is_augmented: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred_mask: [B, 1, H, W] predicted swap mask in [0, 1]
            gt_mask: [B, 1, H, W] ground truth occlusion mask (only for augmented samples)
            is_augmented: [B] bool tensor indicating which samples have synthetic occlusion
        Returns:
            dict with 'mask_bce', 'mask_reg', 'mask_tv' losses
        """
        losses = {}
        device = pred_mask.device

        # BCE loss on augmented samples with known GT
        if gt_mask is not None and is_augmented is not None and is_augmented.any():
            aug_mask = is_augmented.view(-1, 1, 1, 1)
            # Only compute BCE on augmented samples
            bce = F.binary_cross_entropy(
                pred_mask * aug_mask.float(),
                gt_mask * aug_mask.float(),
                reduction="sum",
            ) / (aug_mask.float().sum() * pred_mask.shape[2] * pred_mask.shape[3] + 1e-8)
            losses["mask_bce"] = bce * self.bce_weight
        else:
            losses["mask_bce"] = torch.tensor(0.0, device=device)

        # Regularization: on non-augmented samples, mask should be ~1.0 (swap everything)
        if is_augmented is not None and (~is_augmented).any():
            clean_mask = (~is_augmented).view(-1, 1, 1, 1).float()
            reg = F.l1_loss(
                pred_mask * clean_mask,
                torch.ones_like(pred_mask) * clean_mask,
                reduction="sum",
            ) / (clean_mask.sum() * pred_mask.shape[2] * pred_mask.shape[3] + 1e-8)
            losses["mask_reg"] = reg * self.reg_weight
        else:
            # If no augmentation info, regularize all toward 1.0
            losses["mask_reg"] = F.l1_loss(pred_mask, torch.ones_like(pred_mask)) * self.reg_weight

        # Total variation smoothness
        tv_h = (pred_mask[:, :, 1:, :] - pred_mask[:, :, :-1, :]).pow(2).mean()
        tv_w = (pred_mask[:, :, :, 1:] - pred_mask[:, :, :, :-1]).pow(2).mean()
        losses["mask_tv"] = (tv_h + tv_w) * self.tv_weight

        return losses
