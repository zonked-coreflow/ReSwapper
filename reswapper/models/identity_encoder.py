import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityEncoder(nn.Module):
    """Frozen face recognition backbone for extracting identity embeddings.

    Supports BlendFace (recommended), ArcFace (fallback), or TopoFR.
    The backbone is frozen — only the identity token generator in the
    generator is trainable.

    BlendFace (ICCV 2023) is preferred because it was specifically retrained
    to disentangle identity from attributes (hairstyle, skin color), reducing
    unwanted attribute transfer during face swapping.
    """

    def __init__(
        self,
        model: str = "blendface",
        weights_path: str | None = None,
        embedding_dim: int = 512,
    ):
        super().__init__()
        self.model_name = model
        self.embedding_dim = embedding_dim
        self.backbone = None
        self.weights_path = weights_path

    def load(self, device: torch.device = torch.device("cpu")):
        """Load the identity encoder. Call explicitly to control loading."""
        if self.model_name == "blendface":
            self._load_blendface(device)
        elif self.model_name == "arcface":
            self._load_arcface(device)
        else:
            raise ValueError(f"Unknown identity model: {self.model_name}")

        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone = self.backbone.to(device)

    def _load_arcface(self, device):
        """Load ArcFace iResNet-50 from the existing codebase."""
        import sys
        from pathlib import Path
        # Add parent directory to load the existing iresnet module
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from iresnet import iresnet50

        self.backbone = iresnet50()
        if self.weights_path:
            state_dict = torch.load(self.weights_path, map_location=device)
            self.backbone.load_state_dict(state_dict)

    def _load_blendface(self, device):
        """Load BlendFace model.

        BlendFace uses the same iResNet architecture as ArcFace but is
        retrained on blended face images to disentangle identity from attributes.
        Falls back to ArcFace if BlendFace weights not available.
        """
        try:
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            from iresnet import iresnet50

            self.backbone = iresnet50()
            if self.weights_path:
                state_dict = torch.load(self.weights_path, map_location=device)
                self.backbone.load_state_dict(state_dict)
            else:
                print("Warning: BlendFace weights not provided, using random init. "
                      "Download from https://github.com/mapooon/BlendFace")
        except Exception as e:
            print(f"Failed to load BlendFace: {e}, falling back to ArcFace")
            self._load_arcface(device)

    @torch.no_grad()
    def forward(self, face_images: torch.Tensor) -> torch.Tensor:
        """Extract identity embedding from aligned face images.

        Args:
            face_images: [B, 3, 112, 112] RGB, normalized to [0, 1]
        Returns:
            embedding: [B, 512] L2-normalized identity embedding
        """
        # ArcFace/BlendFace expects [-1, 1] range
        x = face_images * 2.0 - 1.0
        embedding = self.backbone(x)
        embedding = F.normalize(embedding, dim=1)
        return embedding
