import random
import io

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T


class FaceSwapAugmentation:
    """Photometric augmentations for face swap training.

    Avoids spatial transforms that would break face alignment.
    Focuses on color/texture variations that improve robustness.
    """

    def __init__(
        self,
        horizontal_flip: float = 0.5,
        color_jitter: bool = True,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        random_grayscale: float = 0.1,
        jpeg_quality_range: list[int] = (60, 100),
        jpeg_probability: float = 0.3,
        gaussian_blur_probability: float = 0.1,
        gaussian_noise_std: float = 0.02,
        gaussian_noise_probability: float = 0.2,
        random_erasing: float = 0.05,
    ):
        transforms = []

        if horizontal_flip > 0:
            transforms.append(T.RandomHorizontalFlip(p=horizontal_flip))

        if color_jitter:
            transforms.append(
                T.RandomApply(
                    [T.ColorJitter(brightness, contrast, saturation, hue)],
                    p=0.8,
                )
            )

        if random_grayscale > 0:
            transforms.append(T.RandomGrayscale(p=random_grayscale))

        if gaussian_blur_probability > 0:
            transforms.append(
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
                    p=gaussian_blur_probability,
                )
            )

        if random_erasing > 0:
            transforms.append(T.RandomErasing(p=random_erasing, scale=(0.02, 0.1)))

        self.transform = T.Compose(transforms)
        self.jpeg_probability = jpeg_probability
        self.jpeg_quality_range = jpeg_quality_range
        self.gaussian_noise_std = gaussian_noise_std
        self.gaussian_noise_probability = gaussian_noise_probability

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [3, H, W] float tensor in [0, 1]
        Returns:
            augmented image [3, H, W] in [0, 1]
        """
        image = self.transform(image)

        # JPEG compression simulation
        if random.random() < self.jpeg_probability:
            image = self._jpeg_compress(image)

        # Gaussian noise
        if random.random() < self.gaussian_noise_probability:
            noise = torch.randn_like(image) * self.gaussian_noise_std
            image = (image + noise).clamp(0, 1)

        return image

    def _jpeg_compress(self, image: torch.Tensor) -> torch.Tensor:
        """Simulate JPEG compression artifacts."""
        from torchvision.io import encode_jpeg, decode_image
        quality = random.randint(*self.jpeg_quality_range)
        # Convert to uint8 for JPEG encoding
        img_uint8 = (image * 255).byte()
        try:
            encoded = encode_jpeg(img_uint8, quality=quality)
            decoded = decode_image(encoded)
            return decoded.float() / 255.0
        except Exception:
            return image
