"""Image processing utilities.

Ported from the original ReSwapper Image.py, removing the emap.npy
dependency (replaced by direct BlendFace/ArcFace embeddings).
"""

import cv2
import numpy as np
import torch


def to_tensor(image: np.ndarray, size: int | None = None) -> torch.Tensor:
    """Convert BGR numpy image to RGB tensor [1, 3, H, W] in [0, 1].

    Args:
        image: BGR numpy image [H, W, 3]
        size: optional resize dimension
    Returns:
        tensor: [1, 3, H, W] float32 in [0, 1]
    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if size is not None and (img.shape[0] != size or img.shape[1] != size):
        img = cv2.resize(img, (size, size))
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert RGB tensor [1, 3, H, W] in [0, 1] to BGR numpy.

    Args:
        tensor: [1, 3, H, W] or [3, H, W] float tensor in [0, 1]
    Returns:
        image: [H, W, 3] uint8 BGR
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img = tensor.clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def postprocess_face(face_tensor: torch.Tensor) -> np.ndarray:
    """Convert a generated face tensor to BGR numpy image.

    Args:
        face_tensor: [1, 3, H, W] or [3, H, W] in [0, 1]
    Returns:
        face: [H, W, 3] uint8 BGR
    """
    return to_numpy(face_tensor)


def get_blob(
    aligned_image: np.ndarray,
    input_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Create a normalized blob from an aligned face image.

    Legacy compatibility with the original ReSwapper pipeline.

    Args:
        aligned_image: BGR aligned face [H, W, 3]
        input_size: target (W, H)
    Returns:
        blob: [1, 3, H, W] float32 in [0, 1]
    """
    blob = cv2.dnn.blobFromImage(
        aligned_image, 1.0 / 255.0, input_size, (0, 0, 0), swapRB=True
    )
    return blob
