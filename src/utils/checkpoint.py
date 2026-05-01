"""
Checkpoint Manager
===================
Save/load model state, optimizer state, epoch, and metrics with
best-model tracking and top-K checkpoint retention.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    """Manages model checkpoints with best-model tracking.

    Parameters
    ----------
    checkpoint_dir : str or Path
        Directory for storing checkpoint files.
    model_name : str
        Model identifier used in filenames.
    max_keep : int
        Maximum number of periodic checkpoints to retain (FIFO).
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = "./outputs/checkpoints",
        model_name: str = "model",
        max_keep: int = 5,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.max_keep = max_keep
        self._saved_checkpoints: list[Path] = []
        self.best_metric: float | None = None
        self.best_epoch: int | None = None

    def _build_state(
        self,
        model: torch.nn.Module | dict[str, torch.nn.Module],
        optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer] | None,
        epoch: int,
        metrics: dict[str, float] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a checkpoint state dictionary."""
        state: dict[str, Any] = {"epoch": epoch}

        # Handle single model or dict of models (GAN: G + D)
        if isinstance(model, dict):
            state["model_states"] = {
                k: v.state_dict() for k, v in model.items()
            }
        else:
            state["model_state"] = model.state_dict()

        if optimizer is not None:
            if isinstance(optimizer, dict):
                state["optimizer_states"] = {
                    k: v.state_dict() for k, v in optimizer.items()
                }
            else:
                state["optimizer_state"] = optimizer.state_dict()

        if metrics:
            state["metrics"] = metrics
        if extra:
            state.update(extra)

        return state

    def save(
        self,
        model: torch.nn.Module | dict[str, torch.nn.Module],
        optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer] | None,
        epoch: int,
        metrics: dict[str, float] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Save a periodic checkpoint and enforce the max-keep limit."""
        state = self._build_state(model, optimizer, epoch, metrics, extra)
        path = self.checkpoint_dir / f"{self.model_name}_epoch_{epoch:04d}.pt"
        torch.save(state, path)

        self._saved_checkpoints.append(path)
        while len(self._saved_checkpoints) > self.max_keep:
            old = self._saved_checkpoints.pop(0)
            if old.exists():
                old.unlink()

        return path

    def save_best(
        self,
        model: torch.nn.Module | dict[str, torch.nn.Module],
        optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer] | None,
        epoch: int,
        metric_value: float,
        metrics: dict[str, float] | None = None,
        lower_is_better: bool = True,
    ) -> bool:
        """Save checkpoint only if the tracked metric improved.

        Returns ``True`` if a new best was saved.
        """
        is_better = (
            self.best_metric is None
            or (lower_is_better and metric_value < self.best_metric)
            or (not lower_is_better and metric_value > self.best_metric)
        )

        if is_better:
            self.best_metric = metric_value
            self.best_epoch = epoch
            state = self._build_state(model, optimizer, epoch, metrics)
            state["best_metric"] = metric_value
            path = self.checkpoint_dir / f"{self.model_name}_best.pt"
            torch.save(state, path)
            return True

        return False

    @staticmethod
    def load(
        path: str | Path,
        model: torch.nn.Module | dict[str, torch.nn.Module],
        optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer] | None = None,
        device: torch.device | str = "cpu",
    ) -> dict[str, Any]:
        """Load a checkpoint into model(s) and optional optimizer(s).

        Returns the full state dict for access to epoch, metrics, etc.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(path, map_location=device, weights_only=False)

        # Load model weights
        if isinstance(model, dict):
            for key, mod in model.items():
                mod.load_state_dict(state["model_states"][key])
        else:
            model.load_state_dict(state["model_state"])

        # Load optimizer states
        if optimizer is not None:
            if isinstance(optimizer, dict):
                for key, opt in optimizer.items():
                    if key in state.get("optimizer_states", {}):
                        opt.load_state_dict(state["optimizer_states"][key])
            elif "optimizer_state" in state:
                optimizer.load_state_dict(state["optimizer_state"])

        return state

    def get_best_path(self) -> Path:
        """Return the path to the best checkpoint."""
        return self.checkpoint_dir / f"{self.model_name}_best.pt"

    def get_latest_path(self) -> Path | None:
        """Return the most recent periodic checkpoint, if any."""
        if self._saved_checkpoints:
            return self._saved_checkpoints[-1]
        # Fallback: scan directory
        checkpoints = sorted(self.checkpoint_dir.glob(f"{self.model_name}_epoch_*.pt"))
        return checkpoints[-1] if checkpoints else None
