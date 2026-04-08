import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityLoss(nn.Module):
    """Differentiable identity loss using a frozen face recognition model.

    Computes cosine similarity between identity embeddings of the swapped
    output and the source face. Unlike the original ReSwapper's StyleTransferLoss,
    this does NOT run face detection on the output — it directly feeds the
    already-aligned output through the frozen recognition model.

    This is critical: face detection on generated images is non-differentiable,
    unreliable (detection can fail on artifacts), and extremely slow.
    """

    def __init__(self, identity_encoder: nn.Module):
        super().__init__()
        self.identity_encoder = identity_encoder

    def forward(
        self,
        output_images: torch.Tensor,
        source_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            output_images: [B, 3, H, W] generated face in [0, 1], any resolution
            source_embedding: [B, 512] frozen identity embedding of source face
        Returns:
            loss: scalar, 1 - cosine_similarity (lower = more similar)
        """
        # Resize to ArcFace/BlendFace input size
        if output_images.shape[-1] != 112 or output_images.shape[-2] != 112:
            output_resized = F.interpolate(
                output_images, size=(112, 112), mode="bilinear", align_corners=False
            )
        else:
            output_resized = output_images

        # Extract embedding (backbone is frozen but gradients flow through input)
        output_embedding = self.identity_encoder.backbone(output_resized * 2.0 - 1.0)
        output_embedding = F.normalize(output_embedding, dim=1)

        similarity = F.cosine_similarity(output_embedding, source_embedding, dim=1)
        loss = (1.0 - similarity).mean()
        return loss
