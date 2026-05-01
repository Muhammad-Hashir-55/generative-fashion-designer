# 🎨 Generative Fashion Designer

**Advanced Deep Neural Networks Project — Generative AI for Fashion Design**

A production-grade, modular deep learning system implementing **6 advanced generative algorithms** trained on Fashion-MNIST, with comprehensive data pipelines, TensorBoard logging, quantitative evaluation metrics, and professional visualization.

---

## 🏗️ Architecture

```
generative-fashion-designer/
├── config/default.yaml          # Central hyperparameter configuration
├── src/
│   ├── data/                    # Data pipeline (dataset, augmentation, dataloader)
│   ├── models/                  # 6 model architectures
│   │   ├── components.py        # Shared blocks (ResBlock, SelfAttention, SE, etc.)
│   │   ├── vae.py               # β-Variational Autoencoder
│   │   ├── dcgan.py             # Deep Convolutional GAN
│   │   ├── wgan_gp.py           # Wasserstein GAN + Gradient Penalty
│   │   ├── conditional_gan.py   # Class-Conditional GAN (Projection Discriminator)
│   │   ├── style_transfer.py    # VGG-19 Neural Style Transfer
│   │   └── fusion_generator.py  # CVAE-GAN Hybrid Fusion Model
│   ├── training/                # Training engines with early stopping, schedulers
│   ├── evaluation/              # FID, IS, SSIM metrics + visualization
│   ├── inference/               # Unified generation & style mixing
│   └── utils/                   # Config, logging, checkpointing
├── scripts/                     # CLI entry points
│   ├── train.py
│   ├── evaluate.py
│   ├── generate.py
│   └── visualize_data.py
└── outputs/                     # Generated during runtime
    ├── checkpoints/
    ├── logs/                    # TensorBoard event files
    ├── generated/               # Sample images per epoch
    └── evaluation/              # Metrics, reports, plots
```

---

## 🧠 Implemented Models

| # | Model | Key Techniques | Reference |
|---|-------|---------------|-----------|
| 1 | **β-VAE** | Residual encoder, KL annealing, reparameterization trick | Kingma & Welling, 2014 |
| 2 | **DCGAN** | Spectral norm, self-attention, minibatch stddev, label smoothing | Radford et al., 2016 |
| 3 | **WGAN-GP** | Gradient penalty (λ=10), 5:1 critic ratio, Wasserstein distance | Gulrajani et al., 2017 |
| 4 | **Conditional GAN** | Projection discriminator, class embeddings | Miyato & Koyama, 2018 |
| 5 | **VGG-19 NST** | Gram matrix style loss, total variation, L-BFGS optimization | Gatys et al., 2015 |
| 6 | **CVAE-GAN Fusion** | Triple loss (recon + KL + adversarial), VAE encoder + GAN decoder | Larsen et al., 2016 |

### Advanced Components Used Across Models
- **Self-Attention Layers** (SA-GAN, Zhang et al., 2019)
- **Squeeze-and-Excitation** channel attention (Hu et al., 2018)
- **Spectral Normalization** for discriminator stability
- **Minibatch Standard Deviation** for mode collapse detection
- **Residual Blocks** with configurable normalization
- **Gradient Penalty** for Lipschitz constraint enforcement

---

## 📊 Dataset

**Fashion-MNIST** — 70,000 grayscale images (28×28 → upscaled to 32×32)

| Class | Samples |
|-------|---------|
| T-shirt/top | 7,000 |
| Trouser | 7,000 |
| Pullover | 7,000 |
| Dress | 7,000 |
| Coat | 7,000 |
| Sandal | 7,000 |
| Shirt | 7,000 |
| Sneaker | 7,000 |
| Bag | 7,000 |
| Ankle boot | 7,000 |

Auto-downloaded via `torchvision.datasets.FashionMNIST`.

---

## 🚀 Quick Start

### 1. Visualize the Dataset
```bash
python scripts/visualize_data.py
```

### 2. Train a Model
```bash
# Train VAE (50 epochs)
python scripts/train.py --model vae --epochs 50

# Train DCGAN (80 epochs)
python scripts/train.py --model dcgan --epochs 80

# Train WGAN-GP
python scripts/train.py --model wgan_gp --epochs 80

# Train Conditional GAN
python scripts/train.py --model cgan --epochs 80

# Train CVAE-GAN Fusion
python scripts/train.py --model fusion --epochs 60
```

### 3. Generate Samples
```bash
# Generate from VAE
python scripts/generate.py --model vae --num 64

# Generate specific class (cGAN)
python scripts/generate.py --model cgan --class "Dress" --num 16

# Latent space interpolation
python scripts/generate.py --model vae --interpolate --steps 10
```

### 4. Evaluate Models
```bash
# Evaluate single model
python scripts/evaluate.py --model vae --num-samples 100

# Evaluate all trained models
python scripts/evaluate.py --all --num-samples 100
```

### 5. Monitor Training (TensorBoard)
```bash
tensorboard --logdir outputs/logs
```

---

## ⚙️ Configuration

All hyperparameters are centralized in `config/default.yaml`:

```yaml
models:
  latent_dim: 128
  vae:
    beta: 1.0
    beta_anneal_epochs: 10
  wgan_gp:
    lambda_gp: 10.0
    n_critic: 5

training:
  optimizer:
    lr: 2.0e-4
    betas: [0.5, 0.999]
  gradient_clip: 1.0
  early_stopping:
    patience: 15
```

Override from CLI:
```bash
python scripts/train.py --model vae --lr 0.001 --batch-size 256 --epochs 100
```

---

## 📈 Evaluation Metrics

| Metric | What it Measures | Better |
|--------|-----------------|--------|
| **FID** (Fréchet Inception Distance) | Distributional similarity to real data | Lower ↓ |
| **IS** (Inception Score) | Image quality + diversity | Higher ↑ |
| **SSIM** (Structural Similarity) | Perceptual reconstruction quality | Higher ↑ |

---

## 🛠️ Tech Stack

- **PyTorch 2.x** + TorchVision
- **TensorBoard** for experiment tracking
- **scikit-learn** for t-SNE latent visualization
- **Matplotlib** + Seaborn for publication-quality plots
- **CUDA** (RTX 4050 optimized) with CPU fallback

---

## 📁 Project Structure Details

### Data Pipeline (`src/data/`)
- Stratified train/val/test splits preserving class balance
- 4 augmentation pipelines: Train, Eval, GAN, StyleTransfer
- Configurable DataLoader factory with pin_memory + multi-worker

### Training Infrastructure (`src/training/`)
- Abstract `BaseTrainer` with early stopping, gradient clipping
- TensorBoard scalar/image/histogram logging
- Automatic checkpoint management (best + top-K periodic)
- CosineAnnealingWarmRestarts + LinearWarmupCosineDecay schedulers

### Evaluation (`src/evaluation/`)
- Manual FID implementation (eigendecomposition — no external deps)
- Inception Score with configurable splits
- SSIM with Gaussian windowing
- Markdown comparison reports across all models
