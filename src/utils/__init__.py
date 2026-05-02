"""Utility modules: configuration, logging, checkpointing."""

from src.utils.config import load_config, Config
from src.utils.checkpoint import CheckpointManager

# TrainingLogger is imported lazily to avoid pulling in tensorboard/TF at import time
def __getattr__(name: str):
    if name == "TrainingLogger":
        from src.utils.logger import TrainingLogger
        return TrainingLogger
    raise AttributeError(f"module 'src.utils' has no attribute {name!r}")

__all__ = ["load_config", "Config", "TrainingLogger", "CheckpointManager"]
