"""
Learning Rate Schedulers
=========================
Composite and custom LR scheduling strategies.
"""

from __future__ import annotations
import math
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineDecay(_LRScheduler):
    """Linear warmup followed by cosine annealing decay.
    
    For the first ``warmup_epochs``, LR increases linearly from 0 to base_lr.
    Then cosine decay to ``eta_min`` over the remaining epochs.
    """

    def __init__(self, optimizer, warmup_epochs: int = 5,
                 total_epochs: int = 100, eta_min: float = 1e-6,
                 last_epoch: int = -1) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return [self.eta_min + (base_lr - self.eta_min) * cosine
                for base_lr in self.base_lrs]


def build_scheduler(optimizer, config) -> _LRScheduler | None:
    """Factory: build scheduler from config."""
    sched_cfg = getattr(config.training, "scheduler", None)
    if sched_cfg is None:
        return None
    stype = getattr(sched_cfg, "type", "cosine_warm_restarts")
    if stype == "cosine_warm_restarts":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=getattr(sched_cfg, "T_0", 10),
            T_mult=getattr(sched_cfg, "T_mult", 2),
            eta_min=getattr(sched_cfg, "eta_min", 1e-6))
    elif stype == "step":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=getattr(sched_cfg, "step_size", 20),
            gamma=getattr(sched_cfg, "gamma", 0.5))
    elif stype == "warmup_cosine":
        return LinearWarmupCosineDecay(
            optimizer, warmup_epochs=getattr(sched_cfg, "warmup_epochs", 5),
            total_epochs=getattr(config.training, "default_epochs", 50),
            eta_min=getattr(sched_cfg, "eta_min", 1e-6))
    return None
