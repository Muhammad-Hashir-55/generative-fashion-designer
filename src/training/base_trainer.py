"""
Abstract Base Trainer
======================
Provides the skeleton for all training loops: setup, train epoch,
validation, checkpointing, early stopping, and TensorBoard logging.
"""

from __future__ import annotations

import abc
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.logger import TrainingLogger
from src.utils.checkpoint import CheckpointManager
from src.utils.config import Config


class EarlyStopping:
    """Stop training when a monitored metric has stopped improving."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best: float | None = None
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        if self.best is None or metric < self.best - self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class BaseTrainer(abc.ABC):
    """Abstract trainer providing common training infrastructure."""

    def __init__(self, config: Config, model_name: str,
                 device: torch.device | None = None) -> None:
        self.config = config
        self.model_name = model_name
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # ── Performance / numerical settings ────────────────────────────
        self._configure_torch_runtime()
        self._configure_amp()

        self.logger = TrainingLogger(
            log_dir=getattr(config.paths, "log_dir", "./outputs/logs"),
            experiment_name=model_name)
        self.ckpt_mgr = CheckpointManager(
            checkpoint_dir=getattr(config.paths, "checkpoint_dir",
                                   "./outputs/checkpoints"),
            model_name=model_name)
        es_cfg = getattr(config.training, "early_stopping", None)
        self.early_stopping = EarlyStopping(
            patience=getattr(es_cfg, "patience", 15),
            min_delta=getattr(es_cfg, "min_delta", 1e-4),
        ) if es_cfg and getattr(es_cfg, "enabled", True) else None

        self.global_step = 0
        self.current_epoch = 0

        # Output dirs
        gen_dir = getattr(config.paths, "generated_dir", "./outputs/generated")
        Path(gen_dir).mkdir(parents=True, exist_ok=True)
        self.generated_dir = Path(gen_dir) / model_name
        self.generated_dir.mkdir(parents=True, exist_ok=True)

    def _configure_torch_runtime(self) -> None:
        """Apply runtime flags that materially affect throughput on CUDA.

        This is intentionally conservative and fully configurable via
        ``config.training``. Defaults favor speed over bitwise determinism.
        """
        train_cfg = getattr(self.config, "training", None)
        if train_cfg is None:
            return

        deterministic = bool(getattr(train_cfg, "deterministic", False))
        cudnn_benchmark = getattr(train_cfg, "cudnn_benchmark", None)
        if cudnn_benchmark is None:
            cudnn_benchmark = (self.device.type == "cuda") and (not deterministic)

        if self.device.type == "cuda":
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

            tf32 = bool(getattr(train_cfg, "tf32", True))
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = tf32
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = tf32
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high" if tf32 else "highest")

    def _configure_amp(self) -> None:
        """Configure AMP/autocast + GradScaler."""
        train_cfg = getattr(self.config, "training", None)
        mixed_precision = bool(getattr(train_cfg, "mixed_precision", False)) if train_cfg else False

        amp_dtype_str = str(getattr(train_cfg, "amp_dtype", "float16")).lower() if train_cfg else "float16"
        if amp_dtype_str in ("bf16", "bfloat16"):
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16

        self.amp_enabled = bool(mixed_precision and self.device.type == "cuda")
        self.amp_dtype = amp_dtype

        # GradScaler is only needed for fp16. bf16 is typically stable without it.
        use_scaler = self.amp_enabled and (self.amp_dtype == torch.float16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    def autocast(self, enabled: bool | None = None):
        """Return an autocast context manager for the configured device."""
        enabled_final = self.amp_enabled if enabled is None else bool(enabled)
        if not enabled_final:
            return nullcontext()

        # Prefer torch.autocast (works across devices) when available.
        if hasattr(torch, "autocast"):
            return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=True)
        if self.device.type == "cuda":
            return torch.cuda.amp.autocast(enabled=True)
        return nullcontext()

    def backward_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        *,
        clip_params=None,
        clip_value: float | None = None,
        retain_graph: bool = False,
        scaler_update: bool = True,
    ) -> None:
        """Backprop + optional grad clip + optimizer step (AMP-aware)."""
        if self.scaler is not None and self.scaler.is_enabled():
            self.scaler.scale(loss).backward(retain_graph=retain_graph)
            if clip_params is not None and clip_value is not None:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(clip_params, clip_value)
            self.scaler.step(optimizer)
            if scaler_update:
                self.scaler.update()
        else:
            loss.backward(retain_graph=retain_graph)
            if clip_params is not None and clip_value is not None:
                torch.nn.utils.clip_grad_norm_(clip_params, clip_value)
            optimizer.step()

    @abc.abstractmethod
    def setup(self, **kwargs: Any) -> None:
        """Initialize models, optimizers, schedulers."""
        ...

    @abc.abstractmethod
    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Train for one epoch and return metrics dict."""
        ...

    @abc.abstractmethod
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        """Validate and return metrics dict."""
        ...

    @abc.abstractmethod
    def generate_samples(self, epoch: int) -> None:
        """Generate and save sample images."""
        ...

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int) -> dict[str, list[float]]:
        """Full training loop with logging, checkpointing, early stopping."""
        self.logger.log_training_start(self.model_name)
        self.logger.info(f"Device: {self.device} | Epochs: {epochs}")
        history: dict[str, list[float]] = {}

        sample_every = getattr(self.config.training, "sample_every", 5)
        ckpt_every = getattr(self.config.training, "checkpoint_every", 10)

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            t0 = time.time()

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            elapsed = time.time() - t0
            all_metrics = {**{f"train/{k}": v for k, v in train_metrics.items()},
                           **{f"val/{k}": v for k, v in val_metrics.items()}}

            # Log to TensorBoard
            for tag, val in all_metrics.items():
                self.logger.log_scalar(tag, val, epoch)

            # Console output
            display = {k: v for k, v in train_metrics.items()}
            display["time"] = elapsed
            self.logger.log_epoch(epoch, epochs, display)

            # Track history
            for k, v in all_metrics.items():
                history.setdefault(k, []).append(v)

            # Generate samples
            if epoch % sample_every == 0 or epoch == 1:
                self.generate_samples(epoch)

            # Checkpointing
            if epoch % ckpt_every == 0:
                self._save_checkpoint(epoch, train_metrics)

            # Early stopping on first train metric
            if self.early_stopping:
                first_metric = next(iter(train_metrics.values()))
                if self.early_stopping(first_metric):
                    self.logger.info(
                        f"Early stopping triggered at epoch {epoch}")
                    break

        self._save_checkpoint(epoch, train_metrics)
        self.logger.info("Training complete.")
        self.logger.close()
        return history

    def _save_checkpoint(self, epoch: int,
                         metrics: dict[str, float]) -> None:
        """Override in subclass for multi-model saves."""
        pass
