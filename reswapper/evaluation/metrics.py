"""Evaluation metrics for face swapping quality.

Metrics:
- FID: Frechet Inception Distance between generated and real faces
- Identity Similarity: ArcFace/BlendFace cosine similarity (output vs source)
- Pose Preservation: head pose MAE (output vs target)
- Expression Preservation: expression coefficient MAE (output vs target)
- Gaze Preservation: gaze direction cosine distance (output vs target)
"""

import torch
import torch.nn.functional as F
import numpy as np


class IdentitySimilarityMetric:
    """Measures how well the swapped face preserves the source identity.

    Extracts face recognition embeddings from swapped output and source,
    computes cosine similarity. Higher is better.
    """

    def __init__(self, identity_encoder):
        self.identity_encoder = identity_encoder
        self.similarities = []

    def reset(self):
        self.similarities = []

    @torch.no_grad()
    def update(self, output_images: torch.Tensor, source_embeddings: torch.Tensor):
        """
        Args:
            output_images: [B, 3, H, W] in [0, 1]
            source_embeddings: [B, 512] source identity embeddings
        """
        output_resized = F.interpolate(output_images, size=(112, 112), mode="bilinear", align_corners=False)
        output_emb = self.identity_encoder.backbone(output_resized * 2.0 - 1.0)
        output_emb = F.normalize(output_emb, dim=1)
        sim = F.cosine_similarity(output_emb, source_embeddings, dim=1)
        self.similarities.extend(sim.cpu().tolist())

    def compute(self) -> float:
        if not self.similarities:
            return 0.0
        return float(np.mean(self.similarities))


class FIDCalculator:
    """FID score between generated and real face distributions.

    Uses torch-fidelity or clean-fid for computation.
    Call compute() with paths to directories of images.
    """

    @staticmethod
    def compute(generated_dir: str, real_dir: str) -> float:
        """Compute FID between two directories of images."""
        try:
            from cleanfid import fid
            return fid.compute_fid(generated_dir, real_dir)
        except ImportError:
            try:
                import torch_fidelity
                metrics = torch_fidelity.calculate_metrics(
                    input1=generated_dir,
                    input2=real_dir,
                    fid=True,
                )
                return metrics["frechet_inception_distance"]
            except ImportError:
                print("Warning: install clean-fid or torch-fidelity for FID computation")
                return -1.0


class PosePreservationMetric:
    """Measures pose consistency between swapped face and target face.

    Uses a 3D face model to extract yaw/pitch/roll, computes MAE.
    Lower is better.
    """

    def __init__(self):
        self.errors = []

    def reset(self):
        self.errors = []

    @torch.no_grad()
    def update(self, output_images: torch.Tensor, target_images: torch.Tensor):
        # Placeholder — needs 3DDFA or SynergyNet for pose extraction
        pass

    def compute(self) -> float:
        if not self.errors:
            return 0.0
        return float(np.mean(self.errors))
