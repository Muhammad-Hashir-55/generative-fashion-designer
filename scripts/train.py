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

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.utils.config import load_config
from src.data.dataloader import create_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a generative fashion model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["vae", "dcgan", "wgan_gp", "cgan", "fusion"],
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
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    overrides = {}
    if args.batch_size:
        overrides.setdefault("data", {})["batch_size"] = args.batch_size
    if args.lr:
        overrides.setdefault("training", {}).setdefault("optimizer", {})["lr"] = args.lr

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
    else:
        raise ValueError(f"Unknown model: {args.model}")

    trainer.setup()
    history = trainer.fit(loaders["train"], loaders["val"], epochs=epochs)

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


if __name__ == "__main__":
    main()
