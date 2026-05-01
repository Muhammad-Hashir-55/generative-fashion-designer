# Generative Fashion Designer — Advanced DNN Project

## Goal
Build a **complete, production-grade, modular Deep Neural Networks project** for generative fashion design. The system implements **6 advanced generative algorithms** trained on real data, with a professional data pipeline, comprehensive evaluation, TensorBoard logging, and polished inference capabilities.

## Existing State
The repo has a skeleton FastAPI + Next.js scaffold and a VGG-19 style transfer class. The existing code is mostly stubs. We will **restructure the entire project** into a pure DNN codebase (no web app — the focus is heavy DNN engineering).

---

## Architecture Overview

```
generative-fashion-designer/
├── config/                         # YAML configuration files
│   └── default.yaml
├── src/                            # Main source package
│   ├── __init__.py
│   ├── data/                       # Data pipeline
│   │   ├── __init__.py
│   │   ├── dataset.py              # Custom FashionMNIST wrappers
│   │   ├── augmentation.py         # Advanced augmentations
│   │   └── dataloader.py           # DataLoader factory
│   ├── models/                     # All 6 model architectures
│   │   ├── __init__.py
│   │   ├── components.py           # Shared building blocks (ResBlocks, SE, Attention)
│   │   ├── vae.py                  # Variational Autoencoder
│   │   ├── dcgan.py                # Deep Convolutional GAN
│   │   ├── wgan_gp.py              # Wasserstein GAN with Gradient Penalty
│   │   ├── conditional_gan.py      # Class-Conditional GAN (cGAN)
│   │   ├── style_transfer.py       # VGG-19 Neural Style Transfer
│   │   └── fusion_generator.py     # CVAE-GAN Hybrid Fusion Model
│   ├── training/                   # Training engines
│   │   ├── __init__.py
│   │   ├── base_trainer.py         # Abstract base trainer
│   │   ├── vae_trainer.py
│   │   ├── gan_trainer.py          # Shared GAN training logic
│   │   ├── wgan_trainer.py
│   │   ├── cgan_trainer.py
│   │   ├── fusion_trainer.py
│   │   └── scheduler.py           # LR scheduling strategies
│   ├── evaluation/                 # Metrics & evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py              # FID, IS, SSIM, LPIPS (manual impl)
│   │   ├── evaluator.py            # Model evaluation pipeline
│   │   └── visualizer.py           # Rich matplotlib visualizations
│   ├── inference/                  # Generation & style transfer
│   │   ├── __init__.py
│   │   ├── generator.py            # Unified generation interface
│   │   └── style_mixer.py          # NST + generative model fusion
│   └── utils/                      # Utilities
│       ├── __init__.py
│       ├── logger.py               # TensorBoard + console logging
│       ├── checkpoint.py           # Model save/load with metadata
│       └── config.py               # YAML config loader
├── scripts/                        # CLI entry points
│   ├── train.py                    # Train any model via CLI
│   ├── evaluate.py                 # Evaluate trained models
│   ├── generate.py                 # Generate fashion samples
│   └── visualize_data.py           # Visualize dataset & augmentations
├── outputs/                        # Generated during runtime
│   ├── checkpoints/
│   ├── logs/
│   ├── generated/
│   └── evaluation/
├── requirements.txt
└── README.md
```

---

## Dataset

**Fashion-MNIST** (via `torchvision.datasets.FashionMNIST`):
- 70,000 grayscale images (28×28), 10 clothing categories
- Auto-download, no manual setup needed
- We'll upscale to 32×32 for GAN architectures

**10 Classes:** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

---

## Six Advanced Models

### 1. Variational Autoencoder (VAE)
- Convolutional encoder with residual connections
- Reparameterization trick with KL-divergence annealing
- Decoder with transposed convolutions + skip connections
- Loss: Reconstruction (BCE) + β-weighted KL divergence

### 2. Deep Convolutional GAN (DCGAN)
- Generator: Fractional-strided convolutions from latent z → 32×32
- Discriminator: Strided convolutions with spectral normalization
- BatchNorm in Generator, LayerNorm in Discriminator
- Label smoothing + noise injection for training stability

### 3. Wasserstein GAN with Gradient Penalty (WGAN-GP)
- Critic (not discriminator) trained 5× per generator step
- Gradient penalty (λ=10) replaces weight clipping
- No BatchNorm in critic — uses LayerNorm instead
- Wasserstein distance as loss metric

### 4. Conditional GAN (cGAN)
- Class-conditional generation (embed class labels)
- Projection discriminator (Miyato & Koyama)
- Enables targeted generation of specific clothing categories
- Class embedding concatenated to noise vector + fed to discriminator

### 5. VGG-19 Neural Style Transfer (from proposal, enhanced)
- Feature extraction at conv1_1 through conv5_1
- Gram matrix style loss + content loss at conv4_2
- L-BFGS optimization with aggressive VRAM management
- Enhanced: multi-scale style blending, total variation loss

### 6. CVAE-GAN Hybrid Fusion Generator
- Combines VAE's structured latent space with GAN's sharp outputs
- VAE encoder → latent space → GAN generator → discriminator
- Triple loss: Reconstruction + KL + Adversarial
- Produces the highest quality outputs by merging both paradigms

---

## Proposed Changes

### Data Pipeline (`src/data/`)

#### [NEW] [dataset.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/data/dataset.py)
- Custom `FashionDesignerDataset` wrapping torchvision FashionMNIST
- Automatic download + caching in `./data/` directory
- Train/validation/test splits with stratification
- Label-to-name mapping for all 10 categories

#### [NEW] [augmentation.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/data/augmentation.py)
- `TrainAugmentation`: RandomHorizontalFlip, RandomRotation(±10°), RandomAffine, ColorJitter-equivalent for grayscale, random erasing
- `EvalAugmentation`: Resize + Normalize only
- `GANAugmentation`: Resize to 32×32 + normalize to [-1,1] for GAN training

#### [NEW] [dataloader.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/data/dataloader.py)
- Factory function: `create_dataloaders(config) → (train_loader, val_loader, test_loader)`
- Configurable batch size, num_workers, pin_memory
- Class-balanced sampling option for conditional models

---

### Model Architectures (`src/models/`)

#### [NEW] [components.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/models/components.py)
- `ResidualBlock`: Conv → BN → ReLU → Conv → BN + skip
- `SelfAttention`: QKV attention layer for feature maps
- `SpectralNorm` wrapper for discriminator stability
- `ClassEmbedding`: Learnable class embedding for conditional models
- Weight initialization utilities (Xavier, He, Orthogonal)

#### [NEW] [vae.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/models/vae.py)
- `Encoder`: 4 conv layers with residual connections → μ, log_σ²
- `Decoder`: FC → reshape → 4 transposed conv layers → Sigmoid
- `VAE`: Full model with reparameterize(), encode(), decode(), forward()
- Latent dim: 128

#### [NEW] [dcgan.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/models/dcgan.py)
- `DCGenerator`: z(128) → FC → reshape → 4 fractional-strided conv → Tanh
- `DCDiscriminator`: 4 strided conv layers → FC → output
- Spectral normalization on discriminator

#### [NEW] [wgan_gp.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/models/wgan_gp.py)
- `WGANCritic`: No sigmoid, no BatchNorm — LayerNorm + LeakyReLU
- `WGANGenerator`: Same architecture as DCGenerator
- `gradient_penalty()` function

#### [NEW] [conditional_gan.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/models/conditional_gan.py)
- `ConditionalGenerator`: Class embedding concatenated to z → generation
- `ProjectionDiscriminator`: Inner product between class embedding and features
- 10-class conditioning

#### [MODIFY] [style_transfer.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/apps/api/app/ml/style_transfer.py)
- Move to `src/models/style_transfer.py`
- Add total variation loss regularization
- Add multi-resolution style blending
- Add progress callback support

#### [NEW] [fusion_generator.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/models/fusion_generator.py)
- `FusionEncoder`: VAE-style encoder → (μ, logvar)
- `FusionDecoder`: GAN-style generator from latent → image
- `FusionDiscriminator`: Distinguishes real from reconstructed/generated
- `CVAEGANFusion`: Unified model combining all three
- Triple loss computation

---

### Training Engines (`src/training/`)

#### [NEW] [base_trainer.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/training/base_trainer.py)
- Abstract `BaseTrainer` with: setup(), train_epoch(), validate(), fit()
- Automatic checkpointing (best + periodic)
- TensorBoard logging integration
- Gradient clipping, mixed precision support
- EarlyStopping callback

#### [NEW] [vae_trainer.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/training/vae_trainer.py)
- KL annealing schedule (linear warmup over N epochs)
- Reconstruction + KL loss tracking
- Periodic latent space visualization

#### [NEW] [gan_trainer.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/training/gan_trainer.py)
- Two-optimizer setup (G and D)
- Label smoothing (real: 0.9, fake: 0.1)
- Discriminator accuracy tracking
- Fixed noise batch for visual progress tracking

#### [NEW] [wgan_trainer.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/training/wgan_trainer.py)
- Critic-to-generator training ratio (5:1)
- Gradient penalty computation
- Wasserstein distance logging
- Earth Mover's Distance monitoring

#### [NEW] [cgan_trainer.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/training/cgan_trainer.py)
- Class-conditioned training loop
- Per-class generation quality tracking
- Class-balanced batch sampling

#### [NEW] [fusion_trainer.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/training/fusion_trainer.py)
- Triple loss balancing (reconstruction + KL + adversarial)
- Alternating encoder-decoder and discriminator updates
- Dynamic loss weight scheduling

#### [NEW] [scheduler.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/training/scheduler.py)
- CosineAnnealingWarmRestarts wrapper
- LinearWarmup + CosineDecay composite scheduler
- ExponentialDecay with configurable gamma

---

### Evaluation (`src/evaluation/`)

#### [NEW] [metrics.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/evaluation/metrics.py)
- **FID (Fréchet Inception Distance)**: Manual implementation using InceptionV3 features
- **IS (Inception Score)**: KL divergence between conditional and marginal
- **SSIM**: Structural Similarity Index
- **Reconstruction MSE/MAE** for VAE models

#### [NEW] [evaluator.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/evaluation/evaluator.py)
- Unified evaluation pipeline for all models
- Generate N samples → compute all metrics
- Export results as JSON + markdown report
- Comparison table across all 5 generative models

#### [NEW] [visualizer.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/evaluation/visualizer.py)
- Training loss curves (G vs D for GANs)
- Generated sample grids (8×8)
- Latent space t-SNE/UMAP visualization (for VAE)
- Interpolation grids (latent space walks)
- Per-class generation quality grid
- Style transfer before/after comparison

---

### Inference (`src/inference/`)

#### [NEW] [generator.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/inference/generator.py)
- Unified `FashionGenerator` class
- Load any trained model from checkpoint
- Generate single/batch samples
- Latent space interpolation
- Class-conditional generation (for cGAN)

#### [NEW] [style_mixer.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/inference/style_mixer.py)
- Combine generative model output with NST
- Generate base pattern → apply cultural style
- Configurable content/style weight balance

---

### Utilities (`src/utils/`)

#### [NEW] [logger.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/utils/logger.py)
- TensorBoard `SummaryWriter` wrapper
- Console logging with rich formatting
- Scalar, image, histogram logging
- Training progress bars

#### [NEW] [checkpoint.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/utils/checkpoint.py)
- Save/load model state + optimizer state + epoch + metrics
- Best model tracking by metric
- Checkpoint cleanup (keep top-K)

#### [NEW] [config.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/src/utils/config.py)
- YAML configuration loader
- Nested config with dot-notation access
- Default config with override support

---

### Scripts (`scripts/`)

#### [NEW] [train.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/scripts/train.py)
```
python scripts/train.py --model vae --epochs 50 --batch-size 128
python scripts/train.py --model dcgan --epochs 100
python scripts/train.py --model wgan_gp --epochs 100
python scripts/train.py --model cgan --epochs 100
python scripts/train.py --model fusion --epochs 80
```

#### [NEW] [evaluate.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/scripts/evaluate.py)
```
python scripts/evaluate.py --model vae --checkpoint outputs/checkpoints/vae_best.pt
python scripts/evaluate.py --all  # Evaluate all trained models
```

#### [NEW] [generate.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/scripts/generate.py)
```
python scripts/generate.py --model cgan --class "Dress" --num 16
python scripts/generate.py --model fusion --interpolate --steps 10
```

#### [NEW] [visualize_data.py](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/scripts/visualize_data.py)
```
python scripts/visualize_data.py  # Show dataset samples + augmentations
```

---

### Configuration

#### [NEW] [default.yaml](file:///d:/Study/Deep%20Neural%20Networks/generative-fashion-designer/config/default.yaml)
All hyperparameters in one place:
- Data: batch_size, image_size, num_workers
- Model: latent_dim, channels, layers
- Training: lr, betas, epochs, scheduler type
- Evaluation: num_samples, metrics list
- Paths: data_dir, checkpoint_dir, log_dir

---

## Verification Plan

### Automated Tests
1. Run `python scripts/visualize_data.py` — verify dataset loads and augmentations render
2. Run `python scripts/train.py --model vae --epochs 2` — verify VAE trains without errors
3. Run `python scripts/train.py --model dcgan --epochs 2` — verify DCGAN trains
4. Run `python scripts/train.py --model wgan_gp --epochs 2` — verify WGAN-GP trains
5. Run `python scripts/train.py --model cgan --epochs 2` — verify cGAN trains
6. Run `python scripts/train.py --model fusion --epochs 2` — verify Fusion model trains
7. Run `python scripts/generate.py --model vae` — verify generation works
8. Verify TensorBoard logs are created in `outputs/logs/`

### Manual Verification
- Inspect generated sample grids for visual quality
- Check TensorBoard for loss curves
- Verify checkpoint files are saved correctly
