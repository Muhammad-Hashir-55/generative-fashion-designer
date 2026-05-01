"""Lightning training entrypoint for fast multi-GPU VAE training.

This script performs an explicit preflight check first. If the validation
loss exceeds the configured readiness gate, it stops before the expensive
full run begins.
"""

from __future__ import annotations

import os
import warnings
import logging
import argparse
import sys
from pathlib import Path

# Suppress noisy native/runtime logs (oneDNN, absl, TF) and reduce Lightning verbosity.
# These environment variables are recommended by TF/oneDNN to silence repeated INFO messages.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Configure Python logging and warnings early so imported libraries inherit the levels.
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Precision 16-mixed is not supported by the model summary")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from src.data.dataloader import create_dataloaders
from src.training.lightning_vae import LightningVAE
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Lightning VAE trainer")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--devices", type=str, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--force-train", action="store_true", help="Continue into full training even if the readiness gate fails")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _bool_flag(value, default):
    return default if value is None else value


def main():
    args = parse_args()
    overrides = {}
    if args.batch_size is not None:
        overrides.setdefault("data", {})["batch_size"] = args.batch_size
    if args.workers is not None:
        overrides.setdefault("data", {})["num_workers"] = args.workers
    if args.precision is not None:
        overrides.setdefault("training", {}).setdefault("lightning", {})["precision"] = args.precision
    if args.devices is not None:
        overrides.setdefault("training", {}).setdefault("lightning", {})["devices"] = args.devices
    if args.strategy is not None:
        overrides.setdefault("training", {}).setdefault("lightning", {})["strategy"] = args.strategy

    config = load_config(args.config, overrides if overrides else None)
    if args.epochs is not None:
        epochs = args.epochs
    else:
        epochs = int(getattr(config.training, "default_epochs", 50))

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device backend: {device_name}")

    loaders = create_dataloaders(config, mode="train")
    model = LightningVAE(config)

    lightning_cfg = getattr(config.training, "lightning", None)
    if lightning_cfg is None:
        raise ValueError("training.lightning section is required")

    accelerator = getattr(lightning_cfg, "accelerator", "auto")
    devices = getattr(lightning_cfg, "devices", "auto")
    strategy = getattr(lightning_cfg, "strategy", "auto")
    precision = getattr(lightning_cfg, "precision", "16-mixed")
    accumulate_grad_batches = int(getattr(lightning_cfg, "accumulate_grad_batches", 1))
    gradient_clip_val = float(getattr(lightning_cfg, "gradient_clip_val", 1.0))
    deterministic = bool(getattr(lightning_cfg, "deterministic", False))
    benchmark = bool(getattr(lightning_cfg, "benchmark", True))
    log_every_n_steps = int(getattr(lightning_cfg, "log_every_n_steps", 25))

    if accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if devices == "auto":
        devices = torch.cuda.device_count() if accelerator == "gpu" else 1
    if strategy == "auto":
        strategy = "ddp_find_unused_parameters_false" if accelerator == "gpu" and isinstance(devices, int) and devices > 1 else "auto"
    if accelerator != "gpu":
        precision = "32-true"

    if torch.cuda.is_available():
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic

    logger = TensorBoardLogger(
        save_dir=getattr(config.paths, "log_dir", "./outputs/logs"),
        name="vae_lightning",
    )
    ckpt_dir = Path(getattr(config.paths, "checkpoint_dir", "./outputs/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="vae_lightning-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=2,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(monitor="val_loss", mode="min", patience=int(getattr(config.training.early_stopping, "patience", 15))),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_epochs=epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        deterministic=deterministic,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=log_every_n_steps,
        enable_checkpointing=bool(getattr(lightning_cfg, "enable_checkpointing", True)),
        enable_progress_bar=bool(getattr(lightning_cfg, "enable_progress_bar", True)),
    )

    gate = getattr(config.training, "readiness_gate", None)
    if gate and bool(getattr(gate, "enabled", True)):
        min_preflight_epochs = int(getattr(gate, "min_preflight_epochs", 1))
        print(f"Running preflight for {min_preflight_epochs} epoch(s)...")
        preflight = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=precision,
            max_epochs=min_preflight_epochs,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            log_every_n_steps=log_every_n_steps,
            deterministic=deterministic,
        )
        preflight.fit(model, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"], ckpt_path=args.resume)
        metrics = preflight.callback_metrics
        val_recon = float(metrics.get("val_recon", float("inf")))
        val_kl = float(metrics.get("val_kl", float("inf")))
        max_val_recon_loss = float(getattr(gate, "max_val_recon_loss", 0.20))
        max_val_kl_loss = float(getattr(gate, "max_val_kl_loss", 20.0))
        ok = torch.isfinite(torch.tensor(val_recon)) and torch.isfinite(torch.tensor(val_kl)) and val_recon <= max_val_recon_loss and val_kl <= max_val_kl_loss
        print(f"Preflight metrics: val/recon={val_recon:.6f}, val/kl={val_kl:.6f}")
        if not ok:
            message = (
                f"Readiness gate failed: requires val/recon <= {max_val_recon_loss} and val/kl <= {max_val_kl_loss}."
            )
            if args.force_train:
                print(message)
                print("Continuing because --force-train was set.")
            else:
                raise SystemExit(message)
        if args.preflight_only:
            print("Preflight passed. Exiting because --preflight-only was set.")
            return

    trainer.fit(model, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"], ckpt_path=args.resume)


if __name__ == "__main__":
    main()