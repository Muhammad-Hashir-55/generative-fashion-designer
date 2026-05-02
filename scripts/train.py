"""
Training Script
================
CLI entry point for training any generative model.

Usage:
    python scripts/train.py --model vae --epochs 50
    python scripts/train.py --model dcgan --epochs 80
    python scripts/train.py --model wgan_gp --epochs 80
    python scripts/train.py --model cgan --epochs 80
    python scripts/train.py --model fusion --epochs 60
"""

import os
import sys
import argparse
import warnings
import logging
from pathlib import Path

# Quiet noisy native/TensorFlow/oneDNN logs and reduce Python warnings during training runs.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.utils.config import load_config
from src.data.dataloader import create_dataloaders
from src.utils.checkpoint import CheckpointManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a generative fashion model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["vae", "dcgan", "wgan_gp", "cgan", "fusion", "ddpm"],
        help="Model architecture to train",
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cuda/cpu)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume from (or 'latest' / 'best')")
    parser.add_argument("--mixed-precision", action="store_true",
                        help="Enable AMP mixed precision")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic kernels (slower)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override DataLoader workers")
    return parser.parse_args()


def _resolve_resume_path(config, model_name: str, resume_arg: str):
    ckpt_mgr = CheckpointManager(
        checkpoint_dir=getattr(config.paths, "checkpoint_dir", "./outputs/checkpoints"),
        model_name=model_name,
    )
    if resume_arg == "latest":
        return ckpt_mgr.get_latest_path()
    if resume_arg == "best":
        return ckpt_mgr.get_best_path()
    return Path(resume_arg)


def _resume_trainer_state(trainer, ckpt_path: Path, model_name: str) -> int:
    """Load trainer model/optimizer state and return next epoch to run."""
    if model_name == "vae":
        state = CheckpointManager.load(
            ckpt_path, trainer.model, trainer.optimizer, device=trainer.device)
    elif model_name == "dcgan":
        state = CheckpointManager.load(
            ckpt_path,
            {"generator": trainer.generator, "discriminator": trainer.discriminator},
            {"opt_g": trainer.opt_g, "opt_d": trainer.opt_d},
            device=trainer.device,
        )
    elif model_name == "wgan_gp":
        state = CheckpointManager.load(
            ckpt_path,
            {"generator": trainer.generator, "critic": trainer.critic},
            {"opt_g": trainer.opt_g, "opt_c": trainer.opt_c},
            device=trainer.device,
        )
    elif model_name == "cgan":
        state = CheckpointManager.load(
            ckpt_path,
            {"generator": trainer.generator, "discriminator": trainer.discriminator},
            {"opt_g": trainer.opt_g, "opt_d": trainer.opt_d},
            device=trainer.device,
        )
    elif model_name == "fusion":
        state = CheckpointManager.load(
            ckpt_path,
            {"encoder": trainer.model.encoder, "decoder": trainer.model.decoder,
             "discriminator": trainer.model.discriminator},
            {"opt_enc_dec": trainer.opt_enc_dec, "opt_d": trainer.opt_d},
            device=trainer.device,
        )
    elif model_name == "ddpm":
        state = CheckpointManager.load(
            ckpt_path,
            {"model": trainer.model},
            {"optimizer": trainer.optimizer},
            device=trainer.device,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    last_epoch = int(state.get("epoch", 0))
    return last_epoch + 1


def main():
    run_training(parse_args())


def run_training(args):
    """Run training for a single model using parsed CLI args or a compatible namespace."""
    # Load configuration
    overrides = {}
    if args.batch_size:
        overrides.setdefault("data", {})["batch_size"] = args.batch_size
    if args.lr:
        overrides.setdefault("training", {}).setdefault("optimizer", {})["lr"] = args.lr
    if args.mixed_precision:
        overrides.setdefault("training", {})["mixed_precision"] = True
    if args.deterministic:
        overrides.setdefault("training", {})["deterministic"] = True
        overrides.setdefault("training", {})["cudnn_benchmark"] = False
    if args.workers is not None:
        overrides.setdefault("data", {})["num_workers"] = args.workers

    config = load_config(args.config, overrides if overrides else None)

    # Device
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print(f"  Generative Fashion Designer - Training Pipeline")
    print(f"  Model: {args.model.upper()}")
    print(f"  Device: {device}")
    print(f"{'=' * 60}\n")

    # Seed
    seed = getattr(config.project, "seed", 42)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Data
    mode = "gan" if args.model != "vae" else "train"
    loaders = create_dataloaders(config, mode=mode)
    print(f"  Train batches: {len(loaders['train'])}")
    print(f"  Val batches:   {len(loaders['val'])}")
    print(f"  Test batches:  {len(loaders['test'])}\n")

    # Epochs
    epochs_cfg = getattr(config.training, "epochs", None)
    default_epochs = getattr(config.training, "default_epochs", 50)
    if args.epochs:
        epochs = args.epochs
    elif epochs_cfg:
        epochs = getattr(epochs_cfg, args.model, default_epochs)
    else:
        epochs = default_epochs

    # Select trainer
    if args.model == "vae":
        from src.training.vae_trainer import VAETrainer
        trainer = VAETrainer(config, device)
    elif args.model == "dcgan":
        from src.training.gan_trainer import GANTrainer
        trainer = GANTrainer(config, device)
    elif args.model == "wgan_gp":
        from src.training.wgan_trainer import WGANTrainer
        trainer = WGANTrainer(config, device)
    elif args.model == "cgan":
        from src.training.cgan_trainer import CGANTrainer
        trainer = CGANTrainer(config, device)
    elif args.model == "fusion":
        from src.training.fusion_trainer import FusionTrainer
        trainer = FusionTrainer(config, device)
    elif args.model == "ddpm":
        from src.training.ddpm_trainer import DDPMTrainer
        trainer = DDPMTrainer(config, device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    trainer.setup()

    start_epoch = 1
    if args.resume:
        resume_path = _resolve_resume_path(config, args.model, args.resume)
        if resume_path is None or not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        start_epoch = _resume_trainer_state(trainer, resume_path, args.model)
        print(f"  Resumed from {resume_path} (starting at epoch {start_epoch})")

    train_epochs = max(0, epochs - start_epoch + 1)
    if train_epochs == 0:
        print("  Requested epochs already completed by loaded checkpoint. Exiting.")
        return None

    history = trainer.fit(loaders["train"], loaders["val"], epochs=train_epochs)

    # Plot training curves
    from src.evaluation.visualizer import ResultVisualizer
    viz = ResultVisualizer(
        output_dir=getattr(config.paths, "evaluation_dir", "./outputs/evaluation"))

    viz.plot_training_curves(
        {k.replace("train/", ""): v for k, v in history.items() if k.startswith("train/")},
        model_name=args.model)

    if args.model in ("dcgan", "wgan_gp", "cgan"):
        g_key = "train/g_loss"
        d_key = "train/d_loss" if args.model != "wgan_gp" else "train/c_loss"
        if g_key in history and d_key in history:
            viz.plot_gan_loss_comparison(
                history[g_key], history[d_key], model_name=args.model)

    print(f"\n{'=' * 60}")
    print(f"  Training complete! Check outputs/ for results.")
    print(f"{'=' * 60}\n")
    return history


if __name__ == "__main__":
    main()
