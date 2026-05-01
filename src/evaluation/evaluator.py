"""
Model Evaluator
================
Unified evaluation pipeline: generate samples, compute metrics,
and export results as JSON + markdown report.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from src.evaluation.metrics import FIDScore, InceptionScore, compute_ssim
from src.utils.config import Config


class ModelEvaluator:
    """Evaluate generative models and produce comparison reports."""

    def __init__(self, config: Config,
                 device: torch.device | None = None) -> None:
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        eval_dir = getattr(config.paths, "evaluation_dir", "./outputs/evaluation")
        self.eval_dir = Path(eval_dir)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate_model(
        self,
        model_name: str,
        generate_fn,
        real_loader: DataLoader,
        num_samples: int = 1000,
    ) -> dict[str, Any]:
        """Evaluate a single model.

        Parameters
        ----------
        model_name : str
        generate_fn : callable
            Function that takes (num_samples, device) and returns Tensor.
        real_loader : DataLoader
        num_samples : int

        Returns dict of metric values.
        """
        print(f"\n{'═' * 60}")
        print(f"  Evaluating: {model_name}")
        print(f"{'═' * 60}")

        t0 = time.time()

        # Generate samples
        fake_images = generate_fn(num_samples, self.device)
        # Normalise to [0, 1]
        fake_01 = (fake_images + 1.0) / 2.0 if fake_images.min() < 0 else fake_images

        # Collect real images
        real_images = []
        for imgs, _ in real_loader:
            real_images.append(imgs)
            if sum(r.size(0) for r in real_images) >= num_samples:
                break
        real_batch = torch.cat(real_images, dim=0)[:num_samples]
        real_01 = (real_batch + 1.0) / 2.0 if real_batch.min() < 0 else real_batch

        results: dict[str, Any] = {"model": model_name}

        # FID
        eval_cfg = self.config.evaluation
        if getattr(eval_cfg, "fid", None) and getattr(eval_cfg.fid, "enabled", True):
            print("  Computing FID...")
            fid = FIDScore(self.device)
            # Use smaller batches for memory
            batch_sz = min(100, num_samples)
            results["fid"] = fid.compute(
                real_01[:batch_sz].to(self.device),
                fake_01[:batch_sz].to(self.device))
            print(f"  FID: {results['fid']:.2f}")

        # Inception Score
        if getattr(eval_cfg, "inception_score", None) and getattr(
                eval_cfg.inception_score, "enabled", True):
            print("  Computing Inception Score...")
            iscore = InceptionScore(self.device)
            batch_sz = min(100, num_samples)
            is_mean, is_std = iscore.compute(
                fake_01[:batch_sz].to(self.device),
                splits=getattr(eval_cfg.inception_score, "splits", 10))
            results["is_mean"] = is_mean
            results["is_std"] = is_std
            print(f"  IS: {is_mean:.2f} ± {is_std:.2f}")

        # SSIM (for VAE-like models with paired data)
        if getattr(eval_cfg, "ssim", None) and getattr(eval_cfg.ssim, "enabled", True):
            n_ssim = min(fake_01.size(0), real_01.size(0), 200)
            results["ssim"] = compute_ssim(
                real_01[:n_ssim].to(self.device),
                fake_01[:n_ssim].to(self.device))
            print(f"  SSIM: {results['ssim']:.4f}")

        results["eval_time_sec"] = time.time() - t0

        # Save sample grid
        grid = vutils.make_grid(fake_01[:64], nrow=8, padding=2)
        grid_path = self.eval_dir / f"{model_name}_samples.png"
        vutils.save_image(grid, grid_path)

        # Save JSON results
        json_path = self.eval_dir / f"{model_name}_metrics.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"  Results saved to {json_path}")
        return results

    def generate_comparison_report(
        self, all_results: list[dict[str, Any]],
    ) -> str:
        """Generate a markdown comparison table across models."""
        report = "# Generative Model Comparison Report\n\n"
        report += "| Model | FID ↓ | IS ↑ | SSIM ↑ | Time (s) |\n"
        report += "|-------|-------|------|--------|----------|\n"

        for r in all_results:
            fid = f"{r.get('fid', 'N/A'):.2f}" if isinstance(r.get('fid'), float) else "N/A"
            is_val = f"{r.get('is_mean', 'N/A'):.2f}" if isinstance(r.get('is_mean'), float) else "N/A"
            ssim = f"{r.get('ssim', 'N/A'):.4f}" if isinstance(r.get('ssim'), float) else "N/A"
            t = f"{r.get('eval_time_sec', 0):.1f}"
            report += f"| {r['model']} | {fid} | {is_val} | {ssim} | {t} |\n"

        report_path = self.eval_dir / "comparison_report.md"
        report_path.write_text(report, encoding="utf-8")
        print(f"\nComparison report saved to {report_path}")
        return report
