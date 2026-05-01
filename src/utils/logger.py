"""
Training Logger
================
Wraps TensorBoard SummaryWriter with structured console output
and convenience helpers for scalar, image, and histogram logging.
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils


class TrainingLogger:
    """Unified logging to TensorBoard + formatted console output.

    Parameters
    ----------
    log_dir : str or Path
        Root directory for TensorBoard event files.
    experiment_name : str
        Sub-directory name (e.g., ``vae_run_001``).
    console_level : int
        Python logging level for console output.
    """

    def __init__(
        self,
        log_dir: str | Path = "./outputs/logs",
        experiment_name: str | None = None,
        console_level: int = logging.INFO,
    ) -> None:
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"run_{timestamp}"

        self.log_path = Path(log_dir) / experiment_name
        self.log_path.mkdir(parents=True, exist_ok=True)

        # ── TensorBoard writer ────────────────────────────────────────────
        self.writer = SummaryWriter(log_dir=str(self.log_path))

        # ── Console logger ────────────────────────────────────────────────
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(console_level)
        self.logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="│ %(asctime)s │ %(levelname)-8s │ %(message)s",
            datefmt="%H:%M:%S",
        )

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(console_level)
        console.setFormatter(formatter)
        self.logger.addHandler(console)

        # File handler
        file_handler = logging.FileHandler(
            self.log_path / "training.log", encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    # ── Scalar Logging ────────────────────────────────────────────────────
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar value."""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, values: dict[str, float], step: int) -> None:
        """Log multiple scalars under one group."""
        self.writer.add_scalars(main_tag, values, step)

    # ── Image Logging ─────────────────────────────────────────────────────
    def log_images(
        self,
        tag: str,
        images: torch.Tensor,
        step: int,
        nrow: int = 8,
        normalize: bool = True,
    ) -> None:
        """Log a batch of images as a grid."""
        grid = vutils.make_grid(images, nrow=nrow, normalize=normalize, padding=2)
        self.writer.add_image(tag, grid, step)

    # ── Histogram Logging ─────────────────────────────────────────────────
    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """Log a histogram of tensor values."""
        self.writer.add_histogram(tag, values, step)

    # ── Model Graph ───────────────────────────────────────────────────────
    def log_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> None:
        """Log model computational graph."""
        try:
            self.writer.add_graph(model, input_tensor)
        except Exception as exc:
            self.logger.warning(f"Could not log model graph: {exc}")

    # ── Console Methods ───────────────────────────────────────────────────
    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        metrics: dict[str, float],
    ) -> None:
        """Pretty-print epoch summary with metrics."""
        metrics_str = " │ ".join(f"{k}: {v:.6f}" for k, v in metrics.items())
        self.info(f"Epoch [{epoch:>4d}/{total_epochs}] │ {metrics_str}")

    def log_training_start(self, model_name: str, config: Any = None) -> None:
        """Print a styled training header."""
        border = "═" * 70
        self.info(f"╔{border}╗")
        self.info(f"║  Training: {model_name:<57s} ║")
        self.info(f"╚{border}╝")
        if config:
            self.info(f"  Config: {config}")

    # ── Cleanup ───────────────────────────────────────────────────────────
    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        self.writer.flush()
        self.writer.close()

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
