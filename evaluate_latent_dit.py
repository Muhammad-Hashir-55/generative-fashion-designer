"""
Evaluate Latent DiT Model
=========================
Generates metrics specifically for the new diffusion transformer model.
"""

import torch
from src.utils.config import Config
from src.evaluation.evaluator import ModelEvaluator
from src.inference.generator import FashionGenerator
from src.data.dataloader import create_dataloaders

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Starting evaluation of Latent DiT on {device}...")
    
    # 1. Setup Generator
    gen = FashionGenerator(config, model_type="latent_dit", device=device)
    gen.load_checkpoint("outputs/checkpoints/latent_dit_best.pt")
    
    # 2. Setup Evaluator
    evaluator = ModelEvaluator(config, device=device)
    
    # 3. Get Real Data for Comparison
    loaders = create_dataloaders(config, mode="eval")
    dataloader = loaders["test"]
    
    # 4. Run Evaluation
    def generate_fn(n, dev):
        return gen.generate(num_samples=n)
        
    results = evaluator.evaluate_model(
        model_name="latent_dit",
        generate_fn=generate_fn,
        real_loader=dataloader,
        num_samples=100 # Low samples for speed in this environment
    )
    
    print("\nLatent DiT Metrics:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
