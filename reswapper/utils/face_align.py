"""Face alignment utilities.

Ported from the original ReSwapper face_align.py with support for
512px and 1024px resolutions (original only supported 112/128).
"""

import cv2
import numpy as np
from skimage import transform as trans


# ArcFace standard 5-point landmark positions (for 112x112)
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def estimate_norm(landmarks: np.ndarray, image_size: int = 112) -> np.ndarray:
    """Estimate similarity transform from 5-point landmarks to canonical face.

    Args:
        landmarks: [5, 2] facial landmarks (eyes, nose, mouth corners)
        image_size: target crop size (112, 128, 256, 512, 1024)
    Returns:
        M: [2, 3] affine transformation matrix
    """
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio

    dst = ARCFACE_DST * ratio
    dst[:, 0] += diff_x

    if image_size != 128:
        offset = (128 / 32768) * image_size - 0.5
        dst[:, 0] += offset
        dst[:, 1] += offset

    tform = trans.SimilarityTransform()
    tform.estimate(landmarks, dst)
    M = tform.params[0:2, :]
    return M


def norm_crop(
    img: np.ndarray,
    landmarks: np.ndarray,
    image_size: int = 112,
) -> np.ndarray:
    """Align and crop a face using 5-point landmarks.

    Args:
        img: BGR image
        landmarks: [5, 2] facial landmarks
        image_size: target crop size
    Returns:
        warped: aligned face crop [image_size, image_size, 3]
    """
    M = estimate_norm(landmarks, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def norm_crop2(
    img: np.ndarray,
    landmarks: np.ndarray,
    image_size: int = 112,
) -> tuple[np.ndarray, np.ndarray]:
    """Align and crop a face, also returning the transform matrix.

    Args:
        img: BGR image
        landmarks: [5, 2] facial landmarks
        image_size: target crop size
    Returns:
        warped: aligned face crop [image_size, image_size, 3]
        M: [2, 3] affine matrix for inverse warping back
    """
    M = estimate_norm(landmarks, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped, M
