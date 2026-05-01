"""
Generation Script
==================
Generate fashion samples from trained models.

Usage:
    python scripts/generate.py --model vae --num 16
    python scripts/generate.py --model cgan --class "Dress" --num 8
    python scripts/generate.py --model vae --interpolate --steps 10
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torchvision.utils as vutils
from src.utils.config import load_config
from src.inference.generator import FashionGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Generate fashion images")
    parser.add_argument("--model", type=str, required=True,
                        choices=["vae", "dcgan", "wgan_gp", "cgan", "fusion"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num", type=int, default=64, help="Number of samples")
    parser.add_argument("--class", dest="class_label", type=str, default=None,
                        help="Class name for conditional generation")
    parser.add_argument("--interpolate", action="store_true",
                        help="Generate latent interpolation")
    parser.add_argument("--steps", type=int, default=10,
                        help="Interpolation steps")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(getattr(config.paths, "checkpoint_dir", "./outputs/checkpoints"))
    ckpt_path = Path(args.checkpoint) if args.checkpoint else ckpt_dir / f"{args.model}_best.pt"

    gen = FashionGenerator(config, model_type=args.model, device=device)

    if ckpt_path.exists():
        gen.load_checkpoint(ckpt_path)
    else:
        print(f"  Warning: No checkpoint at {ckpt_path}, using random weights")

    output_dir = Path(args.output or getattr(config.paths, "generated_dir", "./outputs/generated"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.interpolate:
        print(f"Generating {args.steps}-step interpolation...")
        images = gen.interpolate(steps=args.steps)
        grid = vutils.make_grid(images, nrow=args.steps, normalize=True, padding=2)
        path = output_dir / f"{args.model}_interpolation.png"
        vutils.save_image(grid, path)
        print(f"Saved to {path}")
    else:
        label = args.class_label
        print(f"Generating {args.num} samples" +
              (f" of class '{label}'" if label else "") + "...")
        images = gen.generate(args.num, class_label=label)
        nrow = min(8, args.num)
        grid = vutils.make_grid(images, nrow=nrow, normalize=True, padding=2)
        suffix = f"_{label.lower().replace('/', '_')}" if label else ""
        path = output_dir / f"{args.model}_generated{suffix}.png"
        vutils.save_image(grid, path)
        print(f"Saved {args.num} samples to {path}")


if __name__ == "__main__":
    main()
