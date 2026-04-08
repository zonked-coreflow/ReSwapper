"""End-to-end face swapping inference pipeline.

Pipeline:
1. InsightFace detection + 5-point landmark alignment
2. SegFace face parsing → 19-class map
3. FLUX VAE encode → 16ch latent
4. BlendFace identity embedding → 16 identity tokens
5. Generator forward → swapped latent + swap mask
6. Latent compositing: mask * swapped + (1-mask) * target
7. FLUX VAE decode → pixel space
8. Poisson blending → paste back into original image
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class FaceSwapper:
    """High-level face swapping interface."""

    def __init__(
        self,
        generator,
        vae,
        identity_encoder,
        face_parser,
        device: torch.device = torch.device("cuda"),
        resolution: int = 512,
    ):
        self.generator = generator
        self.vae = vae
        self.identity_encoder = identity_encoder
        self.face_parser = face_parser
        self.device = device
        self.resolution = resolution

        # Face detection
        from insightface.app import FaceAnalysis
        self.face_analysis = FaceAnalysis(name="buffalo_l")
        self.face_analysis.prepare(ctx_id=0, det_size=(512, 512))

    def swap(
        self,
        target_image: np.ndarray,
        source_image: np.ndarray,
        paste_back: bool = True,
    ) -> np.ndarray:
        """Swap the face in target_image with the identity from source_image.

        Args:
            target_image: BGR image containing the target face
            source_image: BGR image containing the source identity
            paste_back: if True, blend result back into target_image
        Returns:
            BGR result image
        """
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        import face_align
        from reswapper.inference.blend import blend_swapped_face

        # Detect faces
        target_faces = self.face_analysis.get(target_image)
        source_faces = self.face_analysis.get(source_image)

        if len(target_faces) == 0 or len(source_faces) == 0:
            return target_image

        target_face = target_faces[0]
        source_face = source_faces[0]

        # Align target face
        aligned_target, M = face_align.norm_crop2(target_image, target_face.kps, self.resolution)

        # Convert to tensor
        target_tensor = self._to_tensor(aligned_target)  # [1, 3, H, W] in [0, 1]

        # Get source identity embedding
        source_aligned, _ = face_align.norm_crop2(source_image, source_face.kps, 112)
        source_tensor = self._to_tensor(source_aligned, size=112)
        source_embedding = self.identity_encoder(source_tensor)  # [1, 512]

        # Get face parsing map
        latent_size = self.resolution // 8  # FLUX VAE 8x compression
        parsing_map = self.face_parser(target_tensor, target_size=(latent_size, latent_size))

        # VAE encode
        target_latent = self.vae.encode(target_tensor)

        # Generator forward
        with torch.no_grad():
            swapped_latent, swap_mask = self.generator(target_latent, source_embedding, parsing_map)

        # Composite in latent space
        output_latent = swap_mask * swapped_latent + (1 - swap_mask) * target_latent

        # VAE decode
        output_pixels = self.vae.decode(output_latent)

        # Convert to numpy
        swapped_face = self._to_numpy(output_pixels)

        if paste_back:
            # Get swap mask in pixel space for blending
            mask_np = swap_mask.squeeze().cpu().numpy()
            mask_resized = cv2.resize(mask_np, (self.resolution, self.resolution))
            return blend_swapped_face(swapped_face, target_image, M, mask_resized)
        else:
            return swapped_face

    def _to_tensor(self, image: np.ndarray, size: int | None = None) -> torch.Tensor:
        """Convert BGR numpy image to RGB tensor [1, 3, H, W] in [0, 1]."""
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if size and (img.shape[0] != size or img.shape[1] != size):
            img = cv2.resize(img, (size, size))
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert RGB tensor [1, 3, H, W] in [0, 1] to BGR numpy."""
        img = tensor.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
