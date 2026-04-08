"""R3GAN adversarial losses.

Regularized Relativistic Paired GAN loss with R1 and R2 gradient penalties.
This is re-exported from the discriminator module for convenience —
the actual implementations live in reswapper/models/discriminator.py.

Reference: arXiv:2501.05441 "The GAN is dead; long live the GAN!"
"""

from reswapper.models.discriminator import (
    rpgan_loss_d,
    rpgan_loss_g,
    r1_penalty,
    r2_penalty,
)

__all__ = ["rpgan_loss_d", "rpgan_loss_g", "r1_penalty", "r2_penalty"]
