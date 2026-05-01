"""
Comprehensive Training Orchestrator
====================================
Trains all generative models (VAE, DCGAN, WGAN-GP, CGAN, Fusion) sequentially.
All models use GPU acceleration, AMP mixed precision, and transfer learning.

Usage:
    python scripts/train_all_models.py                    # All defaults (50 epochs VAE/fusion, 80 epochs GANs)
    python scripts/train_all_models.py --epochs 100       # Custom epochs
    python scripts/train_all_models.py --models vae dcgan # Specific models only
    python scripts/train_all_models.py --skip-vae         # Skip VAE (or use --skip-dcgan, --skip-wgan, --skip-cgan, --skip-fusion)
"""

import os
import sys
import argparse
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from scripts.train import run_training


class TrainingOrchestrator:
    """Manages sequential training of all models with logging and error handling."""
    
    def __init__(self, args):
        self.args = args
        # Ensure child processes inherit quieting environment variables
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("ABSL_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        logging.getLogger().setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")
        self.python_exe = self._get_python_exe()
        self.project_root = Path(__file__).resolve().parent.parent
        self.log_file = self.project_root / "outputs" / "training_all_models.log"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Model configurations: (name, script, epochs, extra_args)
        self.models = {
            "vae": ("train_lightning.py", args.vae_epochs or 50, ["--force-train", "--precision", "16-mixed"]),
            "dcgan": ("train.py", args.gan_epochs or 80, ["--model", "dcgan", "--mixed-precision"]),
            "wgan_gp": ("train.py", args.gan_epochs or 80, ["--model", "wgan_gp", "--mixed-precision"]),
            "cgan": ("train.py", args.gan_epochs or 80, ["--model", "cgan", "--mixed-precision"]),
            "fusion": ("train.py", args.fusion_epochs or 60, ["--model", "fusion", "--mixed-precision"]),
        }
        
        self.results = {}
        self.start_time = None
        
    def _get_python_exe(self):
        """Get the absolute path to the Python executable."""
        # Use the current interpreter
        return sys.executable
    
    def log(self, message: str):
        """Log message to both console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")
    
    def _should_train_model(self, model_name: str) -> bool:
        """Check if model should be trained based on CLI args."""
        # If specific models are selected, only train those
        if self.args.models:
            return model_name in self.args.models
        
        # Otherwise, check skip flags
        skip_key = f"skip_{model_name}"
        return not getattr(self.args, skip_key, False)
    
    def _build_command(self, model_name: str, script: str, epochs: int, extra_args: list) -> list:
        """Retained for compatibility; batch runner now executes trainers in-process."""
        return [
            self.python_exe,
            str(self.project_root / "scripts" / script),
            "--epochs", str(epochs),
            "--devices", "auto",
            *extra_args,
        ]

    def _build_train_args(self, model_name: str, epochs: int) -> SimpleNamespace:
        """Build a lightweight args namespace for the reusable trainer."""
        return SimpleNamespace(
            model=model_name,
            epochs=epochs,
            batch_size=self.args.batch_size,
            lr=None,
            config=None,
            device=None,
            resume=None,
            mixed_precision=True,
            deterministic=False,
            workers=self.args.workers,
        )
    
    def train_model(self, model_name: str) -> bool:
        """Train a single model and return success status."""
        if not self._should_train_model(model_name):
            self.log(f"⊘ Skipping {model_name.upper()}")
            return True
        
        script, epochs, extra_args = self.models[model_name]
        train_args = self._build_train_args(model_name, epochs)
        
        self.log(f"\n{'='*70}")
        self.log(f"Starting training: {model_name.upper()}")
        self.log(f"Script: {script} | Epochs: {epochs} | GPU: {torch.cuda.is_available()}")
        self.log(f"Training entrypoint: scripts/{script}")
        self.log(f"{'='*70}\n")
        
        try:
            history = run_training(train_args)
            success = history is not None
            self.results[model_name] = {
                "status": "SUCCESS" if success else "FAILED",
                "exit_code": 0 if success else 1,
            }
            
            if success:
                self.log(f"✓ {model_name.upper()} training completed successfully")
            else:
                self.log(f"✗ {model_name.upper()} training failed")
            
            return success
        except Exception as e:
            self.log(f"✗ {model_name.upper()} training error: {str(e)}")
            self.results[model_name] = {"status": "ERROR", "exit_code": -1}
            return False
    
    def run(self):
        """Execute training for all enabled models sequentially."""
        self.start_time = time.time()
        
        if not getattr(self.args, "quiet", False):
            self.log("╔" + "="*68 + "╗")
            self.log("║" + " GENERATIVE FASHION DESIGNER - BATCH TRAINING ".center(68) + "║")
            self.log("║" + " Training all models with GPU acceleration & AMP ".center(68) + "║")
            self.log("╚" + "="*68 + "╝")
        
        self.log(f"Python: {self.python_exe}")
        self.log(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.log(f"GPU Device: {torch.cuda.get_device_name(0)}")
        self.log(f"Project Root: {self.project_root}")
        
        # Determine which models to train
        models_to_train = [m for m in self.models.keys() if self._should_train_model(m)]
        self.log(f"Models to train: {', '.join(m.upper() for m in models_to_train)}\n")
        
        # Train each model sequentially
        success_count = 0
        for model_name in models_to_train:
            if self.train_model(model_name):
                success_count += 1
            self.log("")  # Blank line for readability
        
        # Print summary
        elapsed_time = time.time() - self.start_time
        elapsed_hours = elapsed_time / 3600
        
        self.log("\n" + "="*70)
        self.log("TRAINING SUMMARY")
        self.log("="*70)
        
        for model_name, result in self.results.items():
            status = result["status"]
            symbol = "✓" if status == "SUCCESS" else "✗"
            self.log(f"{symbol} {model_name.upper():15} : {status}")
        
        self.log(f"\nTotal time: {elapsed_hours:.2f} hours ({elapsed_time:.0f} seconds)")
        self.log(f"Success rate: {success_count}/{len(self.results)} models")
        
        if success_count == len(self.results):
            self.log("\n✓ ALL MODELS TRAINED SUCCESSFULLY!")
            self.log(f"Checkpoints saved to: {self.project_root / 'outputs' / 'checkpoints'}")
            self.log(f"Logs saved to: {self.log_file}")
            return 0
        else:
            self.log(f"\n✗ {len(self.results) - success_count} model(s) failed. See log for details.")
            return 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train all generative fashion models sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_all_models.py                          # All models with defaults
  python scripts/train_all_models.py --epochs 100             # All models, 100 epochs each
  python scripts/train_all_models.py --models vae dcgan      # Only VAE and DCGAN
  python scripts/train_all_models.py --skip-vae              # All except VAE
  python scripts/train_all_models.py --vae-epochs 100 --gan-epochs 150  # Custom epochs per type
        """
    )
    
    # Model selection
    parser.add_argument("--models", nargs="+", choices=["vae", "dcgan", "wgan_gp", "cgan", "fusion"],
                        help="Specific models to train (default: all)")
    parser.add_argument("--skip-vae", action="store_true", help="Skip VAE training")
    parser.add_argument("--skip-dcgan", action="store_true", help="Skip DCGAN training")
    parser.add_argument("--skip-wgan", action="store_true", help="Skip WGAN-GP training")
    parser.add_argument("--skip-cgan", action="store_true", help="Skip CGAN training")
    parser.add_argument("--skip-fusion", action="store_true", help="Skip Fusion training")
    parser.add_argument("--quiet", action="store_true", help="Minimal output; suppress banners and tips")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs for all models")
    parser.add_argument("--vae-epochs", type=int, default=None,
                        help="Epochs for VAE (default: 50)")
    parser.add_argument("--gan-epochs", type=int, default=None,
                        help="Epochs for all GANs (default: 80)")
    parser.add_argument("--fusion-epochs", type=int, default=None,
                        help="Epochs for Fusion (default: 60)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override DataLoader workers")
    
    args = parser.parse_args()
    
    # If --epochs is set, override all specific epoch args
    if args.epochs:
        args.vae_epochs = args.epochs
        args.gan_epochs = args.epochs
        args.fusion_epochs = args.epochs
    
    # Handle --skip-wgan (map to skip_wgan_gp)
    if args.skip_wgan:
        args.skip_wgan_gp = True
    else:
        args.skip_wgan_gp = False
    
    return args


def main():
    args = parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Training will use CPU (very slow).")
        print("Proceeding anyway...\n")
    
    orchestrator = TrainingOrchestrator(args)
    exit_code = orchestrator.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
