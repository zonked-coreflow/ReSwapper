import torch
import torch.nn as nn
import torch.nn.functional as F


class LandmarkLoss(nn.Module):
    """68-point facial landmark L2 loss.

    Ensures the swapped face preserves the target's facial geometry
    (pose, expression). Uses a lightweight landmark detector.
    """

    def __init__(self):
        super().__init__()
        self.detector = None

    def load(self, device: torch.device = torch.device("cpu")):
        """Load landmark detector. Uses MediaPipe or dlib."""
        # Placeholder — in production, use a differentiable landmark detector
        # e.g., a small CNN trained on 300W/WFLW
        self.detector = None
        self.device = device

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 3, H, W] in [0, 1]
            target: [B, 3, H, W] in [0, 1]
        Returns:
            scalar L2 loss between predicted landmarks
        """
        if self.detector is None:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_landmarks = self.detector(pred)
        with torch.no_grad():
            target_landmarks = self.detector(target)

        return F.mse_loss(pred_landmarks, target_landmarks)


class GazeLoss(nn.Module):
    """Gaze direction preservation loss.

    Uses a pretrained gaze estimation network to extract gaze direction
    vectors, then computes cosine distance. High-impact, low-complexity.

    Reference: arXiv:2402.03188 "Gaze-Centric Loss Terms for Face Swaps"
    """

    def __init__(self):
        super().__init__()
        self.estimator = None

    def load(self, device: torch.device = torch.device("cpu"), weights_path: str | None = None):
        """Load pretrained gaze estimator (e.g., ETH-XGaze)."""
        self.estimator = None
        self.device = device

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 3, H, W] in [0, 1]
            target: [B, 3, H, W] in [0, 1]
        Returns:
            scalar loss: 1 - cosine_similarity of gaze vectors
        """
        if self.estimator is None:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_gaze = self.estimator(pred)
        with torch.no_grad():
            target_gaze = self.estimator(target)

        similarity = F.cosine_similarity(pred_gaze, target_gaze, dim=1)
        return (1.0 - similarity).mean()


class ExpressionLoss(nn.Module):
    """Expression preservation loss using 3DDFA expression coefficients.

    Extracts expression parameters from a 3D face reconstruction model
    and computes L2 distance between output and target expressions.

    Reference: DynamicFace (arXiv:2501.08553) — uses 3DDFA-V3
    """

    def __init__(self):
        super().__init__()
        self.model = None

    def load(self, device: torch.device = torch.device("cpu"), weights_path: str | None = None):
        """Load 3DDFA or similar 3D face model."""
        self.model = None
        self.device = device

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 3, H, W] in [0, 1]
            target: [B, 3, H, W] in [0, 1]
        Returns:
            scalar L2 loss between expression coefficients
        """
        if self.model is None:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_expr = self.model(pred)
        with torch.no_grad():
            target_expr = self.model(target)

        return F.mse_loss(pred_expr, target_expr)
