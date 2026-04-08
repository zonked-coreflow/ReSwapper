"""Synthetic occlusion augmentation for training the swap mask.

Generates realistic occlusion patterns that the model must learn to
detect and preserve during face swapping:

1. **Tubes/lines** — medical tubes, oxygen cannulas, straws, wires
   crossing the face. Simulated as bezier curves with variable width,
   color, and transparency.

2. **Hands/fingers** — large irregular patches from hand-shaped masks
   or elongated blob clusters simulating finger spreads.

3. **Rectangular/elliptical objects** — microphones, phones, glasses,
   food items held near the face.

4. **Mouth interior / tongue** — special handling: the mouth region
   gets partial mask reduction to teach the model to preserve
   open-mouth expressions and tongue visibility from the target.

References:
    - CelebAMat (FaceMat, arXiv:2508.03055) — multi-source compositing
    - VividFace (arXiv:2412.11279) — comprehensive occlusion augmentation
    - SelfSwapper (arXiv:2402.07370) — perforation confusion
"""

import math
import random

import torch
import torch.nn.functional as F


class SyntheticOcclusionAugmentation:
    """Synthetic occlusion compositing for training the swap mask.

    On ~30% of training steps, overlays synthetic occlusions onto face
    images with known alpha masks. The generator learns to predict
    occluded regions via the mask BCE loss.
    """

    def __init__(
        self,
        probability: float = 0.3,
        num_occlusions_range: tuple[int, int] = (1, 3),
        size_range: tuple[float, float] = (0.05, 0.25),
        tube_probability: float = 0.3,
        hand_probability: float = 0.2,
        mouth_preserve_probability: float = 0.15,
    ):
        self.probability = probability
        self.num_occlusions_range = num_occlusions_range
        self.size_range = size_range
        self.tube_probability = tube_probability
        self.hand_probability = hand_probability
        self.mouth_preserve_probability = mouth_preserve_probability

    def __call__(
        self, image: torch.Tensor, parsing_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Args:
            image: [3, H, W] face image in [0, 1]
            parsing_map: [19, H, W] optional face parsing (for mouth-aware augmentation)
        Returns:
            augmented_image: [3, H, W] with occlusions
            occlusion_mask: [1, H, W] where 1=swap, 0=keep target
            is_augmented: bool
        """
        H, W = image.shape[1], image.shape[2]
        mask = torch.ones(1, H, W, device=image.device)

        if random.random() > self.probability:
            return image, mask, False

        augmented = image.clone()
        num_occlusions = random.randint(*self.num_occlusions_range)

        for _ in range(num_occlusions):
            # Choose occlusion type with weighted probabilities
            r = random.random()
            if r < self.tube_probability:
                augmented, mask = _apply_tube(augmented, mask, H, W)
            elif r < self.tube_probability + self.hand_probability:
                augmented, mask = _apply_hand(augmented, mask, H, W)
            else:
                augmented, mask = _apply_shape(
                    augmented, mask, H, W, self.size_range
                )

        # Mouth/tongue preservation: reduce mask in mouth region to teach
        # the model to be careful around open mouths and tongues
        if parsing_map is not None and random.random() < self.mouth_preserve_probability:
            mask = _apply_mouth_preservation(mask, parsing_map)

        return augmented.clamp(0, 1), mask, True


def _apply_tube(
    image: torch.Tensor, mask: torch.Tensor, H: int, W: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate tubes, wires, cannulas crossing the face.

    Uses quadratic bezier curves with variable width, color, and alpha.
    Tubes typically run horizontally (oxygen cannula under nose) or
    diagonally (medical tubes, straws going to mouth).
    """
    device = image.device

    # Tube parameters
    tube_width = random.randint(max(2, H // 80), max(4, H // 25))  # 2-20px at 512
    tube_color = torch.rand(3, device=device) * 0.5 + 0.2  # mid-tones (tube-like)
    tube_alpha = random.uniform(0.7, 1.0)

    # Choose tube trajectory type
    trajectory = random.choice(["horizontal", "diagonal", "to_mouth", "across_face"])

    if trajectory == "horizontal":
        # Oxygen cannula style — runs under the nose area
        y_center = random.randint(H // 3, 2 * H // 3)
        p0 = (0, y_center + random.randint(-H // 10, H // 10))
        p1 = (W // 2, y_center + random.randint(-H // 15, H // 15))
        p2 = (W, y_center + random.randint(-H // 10, H // 10))

    elif trajectory == "diagonal":
        # Wire or tube running diagonally
        side = random.choice(["tl_br", "tr_bl", "l_r"])
        if side == "tl_br":
            p0 = (random.randint(0, W // 4), 0)
            p1 = (W // 2 + random.randint(-W // 6, W // 6), H // 2)
            p2 = (random.randint(3 * W // 4, W), H)
        elif side == "tr_bl":
            p0 = (random.randint(3 * W // 4, W), 0)
            p1 = (W // 2 + random.randint(-W // 6, W // 6), H // 2)
            p2 = (random.randint(0, W // 4), H)
        else:
            y = random.randint(H // 4, 3 * H // 4)
            p0 = (0, y + random.randint(-H // 8, H // 8))
            p1 = (W // 2, y + random.randint(-H // 6, H // 6))
            p2 = (W, y + random.randint(-H // 8, H // 8))

    elif trajectory == "to_mouth":
        # Straw, tube going into the mouth area (center-bottom of face)
        mouth_x = W // 2 + random.randint(-W // 8, W // 8)
        mouth_y = int(H * 0.7) + random.randint(-H // 10, H // 10)
        # Come from a random edge
        edge = random.choice(["left", "right", "bottom"])
        if edge == "left":
            p0 = (0, random.randint(H // 3, 2 * H // 3))
        elif edge == "right":
            p0 = (W, random.randint(H // 3, 2 * H // 3))
        else:
            p0 = (random.randint(W // 4, 3 * W // 4), H)
        p2 = (mouth_x, mouth_y)
        p1 = ((p0[0] + p2[0]) // 2 + random.randint(-W // 6, W // 6),
              (p0[1] + p2[1]) // 2 + random.randint(-H // 6, H // 6))

    else:  # across_face
        # Full crossing — edge to edge
        side_start = random.choice(["top", "left", "bottom", "right"])
        if side_start == "top":
            p0 = (random.randint(0, W), 0)
            p2 = (random.randint(0, W), H)
        elif side_start == "left":
            p0 = (0, random.randint(0, H))
            p2 = (W, random.randint(0, H))
        elif side_start == "bottom":
            p0 = (random.randint(0, W), H)
            p2 = (random.randint(0, W), 0)
        else:
            p0 = (W, random.randint(0, H))
            p2 = (0, random.randint(0, H))
        p1 = (W // 2 + random.randint(-W // 4, W // 4),
              H // 2 + random.randint(-H // 4, H // 4))

    # Rasterize bezier curve as a soft mask
    tube_mask = _rasterize_bezier_tube(p0, p1, p2, tube_width, H, W, device)
    tube_mask = tube_mask * tube_alpha

    # Apply tube to image
    color_img = tube_color.view(3, 1, 1).expand(3, H, W)
    image = image * (1 - tube_mask) + color_img * tube_mask

    # Update occlusion mask
    mask = mask * (1 - tube_mask)

    return image, mask


def _rasterize_bezier_tube(
    p0: tuple, p1: tuple, p2: tuple,
    width: int, H: int, W: int, device: torch.device,
) -> torch.Tensor:
    """Rasterize a quadratic bezier curve as a soft tube mask.

    Returns [1, H, W] float mask where 1.0 = tube present.
    """
    # Sample points along the bezier curve
    num_samples = max(H, W)
    t = torch.linspace(0, 1, num_samples, device=device)

    # Quadratic bezier: B(t) = (1-t)^2*P0 + 2*(1-t)*t*P1 + t^2*P2
    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]

    # Create distance field from curve points
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )  # [H, W]

    # Compute minimum distance from each pixel to any curve point
    # Process in chunks to avoid OOM
    min_dist = torch.full((H, W), float("inf"), device=device)
    chunk_size = 64
    for i in range(0, num_samples, chunk_size):
        cx = x[i:i + chunk_size].view(-1, 1, 1)  # [chunk, 1, 1]
        cy = y[i:i + chunk_size].view(-1, 1, 1)
        dist = ((xx.unsqueeze(0) - cx) ** 2 + (yy.unsqueeze(0) - cy) ** 2).sqrt()
        chunk_min = dist.min(dim=0).values
        min_dist = torch.min(min_dist, chunk_min)

    # Soft tube mask: 1.0 inside tube, smooth falloff at edges
    half_width = width / 2.0
    tube_mask = (1.0 - (min_dist / half_width).clamp(0, 1)).unsqueeze(0)

    # Smooth edges slightly
    if width > 4:
        tube_mask = _gaussian_blur_2d(tube_mask, kernel_size=3, sigma=0.8)

    return tube_mask


def _apply_hand(
    image: torch.Tensor, mask: torch.Tensor, H: int, W: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate a hand or fingers touching/covering the face.

    Creates elongated blob clusters resembling finger spreads.
    """
    device = image.device

    # Hand color — skin-toned
    skin_base = torch.tensor([0.7, 0.5, 0.4], device=device)  # warm skin tone
    skin_variation = torch.rand(3, device=device) * 0.3 - 0.15
    hand_color = (skin_base + skin_variation).clamp(0, 1)
    hand_alpha = random.uniform(0.85, 1.0)

    # Generate 3-5 "fingers" as elongated ellipses radiating from a point
    num_fingers = random.randint(3, 5)
    # Palm/origin position
    palm_x = random.randint(W // 6, 5 * W // 6)
    palm_y = random.randint(H // 6, 5 * H // 6)

    hand_mask = torch.zeros(1, H, W, device=device)
    base_angle = random.uniform(0, 2 * math.pi)

    for i in range(num_fingers):
        # Each finger is an elongated ellipse
        finger_length = random.randint(H // 6, H // 3)
        finger_width = random.randint(max(3, H // 40), max(6, H // 15))
        angle = base_angle + random.uniform(-0.4, 0.4) + i * (0.2 + random.uniform(-0.05, 0.05))

        # Finger center
        cx = palm_x + int(math.cos(angle) * finger_length * 0.5)
        cy = palm_y + int(math.sin(angle) * finger_length * 0.5)

        # Create rotated ellipse mask
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device) - cy,
            torch.arange(W, dtype=torch.float32, device=device) - cx,
            indexing="ij",
        )
        # Rotate coordinates
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)
        xx_rot = xx * cos_a - yy * sin_a
        yy_rot = xx * sin_a + yy * cos_a

        # Ellipse equation
        ellipse = (xx_rot / (finger_length / 2)) ** 2 + (yy_rot / (finger_width / 2)) ** 2
        finger_mask = (1.0 - ellipse.clamp(0, 1)).clamp(0, 1).unsqueeze(0)
        hand_mask = torch.max(hand_mask, finger_mask)

    # Add palm area (larger circle)
    palm_radius = random.randint(H // 10, H // 5)
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device) - palm_y,
        torch.arange(W, dtype=torch.float32, device=device) - palm_x,
        indexing="ij",
    )
    palm_dist = (xx ** 2 + yy ** 2).sqrt()
    palm_mask = (1.0 - (palm_dist / palm_radius).clamp(0, 1)).unsqueeze(0)
    hand_mask = torch.max(hand_mask, palm_mask)

    # Smooth the hand mask
    hand_mask = _gaussian_blur_2d(hand_mask, kernel_size=7, sigma=2.0)
    hand_mask = hand_mask * hand_alpha

    # Apply
    color_img = hand_color.view(3, 1, 1).expand(3, H, W)
    image = image * (1 - hand_mask) + color_img * hand_mask
    mask = mask * (1 - hand_mask)

    return image, mask


def _apply_shape(
    image: torch.Tensor, mask: torch.Tensor, H: int, W: int,
    size_range: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rectangular, elliptical, or noise-patch occlusion."""
    device = image.device
    occlusion_type = random.choice(["rectangle", "ellipse", "noise_patch"])

    size_frac = random.uniform(*size_range)
    oh = int(H * size_frac)
    ow = int(W * size_frac * random.uniform(0.5, 2.0))
    oh = min(oh, H - 1)
    ow = min(ow, W - 1)
    top = random.randint(0, H - oh)
    left = random.randint(0, W - ow)

    if occlusion_type == "rectangle":
        color = torch.rand(3, 1, 1, device=device)
        image[:, top:top + oh, left:left + ow] = color
        mask[:, top:top + oh, left:left + ow] = 0.0

    elif occlusion_type == "ellipse":
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, oh, device=device),
            torch.linspace(-1, 1, ow, device=device),
            indexing="ij",
        )
        ellipse = (xx ** 2 + yy ** 2) < 1.0
        alpha = ellipse.float().unsqueeze(0)
        if oh > 4 and ow > 4:
            alpha = _gaussian_blur_2d(alpha, kernel_size=5, sigma=1.5)

        color = torch.rand(3, 1, 1, device=device).expand(3, oh, ow)
        image[:, top:top + oh, left:left + ow] = (
            image[:, top:top + oh, left:left + ow] * (1 - alpha) + color * alpha
        )
        mask[:, top:top + oh, left:left + ow] = (
            mask[:, top:top + oh, left:left + ow] * (1 - alpha)
        )

    elif occlusion_type == "noise_patch":
        noise = torch.rand(3, oh, ow, device=device)
        image[:, top:top + oh, left:left + ow] = noise
        mask[:, top:top + oh, left:left + ow] = 0.0

    return image, mask


def _apply_mouth_preservation(
    mask: torch.Tensor, parsing_map: torch.Tensor,
) -> torch.Tensor:
    """Reduce swap confidence in the mouth region to preserve tongues
    and open-mouth expressions from the target.

    The mouth interior (teeth, tongue) is part of the target's expression
    and should NOT be fully replaced with the source's identity. This
    augmentation teaches the mask to be cautious around the mouth.

    Face parser classes used:
        11: mouth (interior)
        12: upper lip
        13: lower lip
    """
    # Get mouth region from parsing map
    # Classes 11 (mouth), 12 (u_lip), 13 (l_lip)
    mouth_region = parsing_map[11] + parsing_map[12] + parsing_map[13]
    mouth_region = mouth_region.clamp(0, 1).unsqueeze(0)  # [1, H, W]

    # Reduce mask in mouth region — don't zero it, just reduce confidence
    # This teaches the model to partially preserve the mouth area
    mouth_reduction = random.uniform(0.2, 0.6)
    mask = mask * (1 - mouth_region * mouth_reduction)

    return mask


def _gaussian_blur_2d(
    x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.5,
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
