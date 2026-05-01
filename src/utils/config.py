"""
YAML Configuration Loader
==========================
Loads hierarchical YAML configs with dot-notation access and default overrides.
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any


class Config:
    """Recursive dictionary wrapper providing dot-notation attribute access.

    Supports nested configs so that ``cfg.models.vae.beta`` works like a
    regular attribute chain rather than ``cfg['models']['vae']['beta']``.
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        data = data or {}
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            elif isinstance(value, list):
                setattr(self, key, [
                    Config(v) if isinstance(v, dict) else v for v in value
                ])
            else:
                setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert back to a plain dictionary."""
        result: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    v.to_dict() if isinstance(v, Config) else v for v in value
                ]
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Safe attribute access with default."""
        return getattr(self, key, default)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"Config({items})"

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (in-place)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load a YAML configuration file and return a ``Config`` object.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to the YAML file.  Defaults to ``config/default.yaml``
        relative to the project root.
    overrides : dict, optional
        Key-value pairs that override loaded values (deep-merged).
    """
    if config_path is None:
        # Walk up from this file to find the project root
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "config" / "default.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)

    if overrides:
        data = _deep_merge(data, overrides)

    return Config(data)
