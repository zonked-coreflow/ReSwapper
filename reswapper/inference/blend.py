import cv2
import numpy as np


def blend_swapped_face(
    swapped_face: np.ndarray,
    target_image: np.ndarray,
    M: np.ndarray,
    swap_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Blend the swapped face back into the original image.

    Uses Poisson blending for color harmonization with soft mask fallback.

    Args:
        swapped_face: [H, W, 3] BGR swapped face (aligned crop)
        target_image: [H_orig, W_orig, 3] BGR original image
        M: [2, 3] affine transformation matrix from alignment
        swap_mask: [H, W] optional swap mask from generator (1=swapped, 0=original)
    Returns:
        result: [H_orig, W_orig, 3] BGR blended image
    """
    h_orig, w_orig = target_image.shape[:2]
    h_face, w_face = swapped_face.shape[:2]

    # Inverse warp the swapped face back to original image space
    M_inv = cv2.invertAffineTransform(M)
    warped_face = cv2.warpAffine(
        swapped_face, M_inv, (w_orig, h_orig), borderMode=cv2.BORDER_REPLICATE
    )

    # Create mask
    if swap_mask is not None:
        # Use the generator's predicted swap mask
        mask_face = (swap_mask * 255).astype(np.uint8)
    else:
        mask_face = np.ones((h_face, w_face), dtype=np.uint8) * 255

    # Warp mask to original space
    warped_mask = cv2.warpAffine(mask_face, M_inv, (w_orig, h_orig))

    # Erode to avoid boundary artifacts
    k = max(int(max(h_face, w_face) / 10), 3)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    warped_mask = cv2.erode(warped_mask, kernel, iterations=1)

    # Gaussian feather the edges
    blur_k = max(int(max(h_face, w_face) / 20), 3)
    if blur_k % 2 == 0:
        blur_k += 1
    warped_mask = cv2.GaussianBlur(warped_mask, (blur_k, blur_k), 0)

    # Try Poisson blending for better color matching
    mask_binary = (warped_mask > 128).astype(np.uint8) * 255
    if mask_binary.sum() > 0:
        # Find center of mask for Poisson blending
        moments = cv2.moments(mask_binary)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            try:
                result = cv2.seamlessClone(
                    warped_face, target_image, mask_binary, (cx, cy), cv2.NORMAL_CLONE
                )
                return result
            except cv2.error:
                pass

    # Fallback: alpha blending
    mask_float = warped_mask.astype(np.float32) / 255.0
    mask_3ch = mask_float[:, :, np.newaxis]
    result = (warped_face * mask_3ch + target_image * (1 - mask_3ch)).astype(np.uint8)
    return result
