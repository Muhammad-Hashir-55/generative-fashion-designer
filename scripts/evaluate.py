"""
Evaluation Script
==================
Evaluate trained models using FID, IS, SSIM metrics.

Usage:
    python scripts/evaluate.py --model vae --checkpoint outputs/checkpoints/vae_best.pt
    python scripts/evaluate.py --all
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.utils.config import load_config
from src.data.dataloader import create_dataloaders
from src.evaluation.evaluator import ModelEvaluator
from src.inference.generator import FashionGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generative models")
    parser.add_argument("--model", type=str, default=None,
                        choices=["vae", "dcgan", "wgan_gp", "cgan", "fusion"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all models with best checkpoints")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples for evaluation")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = create_dataloaders(config, mode="eval")
    evaluator = ModelEvaluator(config, device)

    all_results = []

    if args.all:
        models = ["vae", "dcgan", "wgan_gp", "cgan", "fusion"]
    elif args.model:
        models = [args.model]
    else:
        print("Specify --model or --all")
        return

    ckpt_dir = Path(getattr(config.paths, "checkpoint_dir", "./outputs/checkpoints"))

    for model_name in models:
        ckpt_path = Path(args.checkpoint) if args.checkpoint else ckpt_dir / f"{model_name}_best.pt"
        if not ckpt_path.exists():
            print(f"  Skipping {model_name} — no checkpoint at {ckpt_path}")
            continue

        gen = FashionGenerator(config, model_type=model_name, device=device)
        gen.load_checkpoint(ckpt_path)

        result = evaluator.evaluate_model(
            model_name=model_name,
            generate_fn=lambda n, d: gen.generate(n),
            real_loader=loaders["test"],
            num_samples=args.num_samples)
        all_results.append(result)

    if len(all_results) > 1:
        evaluator.generate_comparison_report(all_results)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
