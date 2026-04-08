import heapq
from pathlib import Path

import torch


class CheckpointManager:
    """Manages saving/loading of training checkpoints with top-K tracking."""

    def __init__(
        self,
        save_dir: str,
        keep_top_k: int = 5,
        metric_mode: str = "max",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.metric_mode = metric_mode
        # Min-heap for max mode (negate values), max-heap for min mode
        self.checkpoints: list[tuple[float, str]] = []

    def save(
        self,
        state: dict,
        step: int,
        metric_value: float | None = None,
    ) -> str:
        path = self.save_dir / f"checkpoint-{step}.pt"
        torch.save(state, path)

        if metric_value is not None:
            score = -metric_value if self.metric_mode == "max" else metric_value
            heapq.heappush(self.checkpoints, (score, str(path)))

            # Remove excess checkpoints
            while len(self.checkpoints) > self.keep_top_k:
                _, old_path = heapq.heappop(self.checkpoints)
                old = Path(old_path)
                if old.exists():
                    old.unlink()

        return str(path)

    def load(self, path: str) -> dict:
        return torch.load(path, map_location="cpu", weights_only=False)

    def get_best_path(self) -> str | None:
        if not self.checkpoints:
            return None
        # Best = largest metric for max mode
        if self.metric_mode == "max":
            return min(self.checkpoints, key=lambda x: x[0])[1]
        return min(self.checkpoints, key=lambda x: x[0])[1]
