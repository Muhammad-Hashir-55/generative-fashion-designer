---
title: Generative Fashion Designer
emoji: 🎨
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
---

# 🎨 Generative Fashion Designer

**Advanced Deep Neural Networks Project — Generative AI for Fashion Design**

A production-grade, modular deep learning system implementing **6 advanced generative architectures**. This project features a full deployment pipeline to Hugging Face Spaces, GPU-optimized training, and cloud-based high-fidelity generation.

---

## 🌐 Live Demo
Check out the interactive web application on Hugging Face Spaces:  
👉 **[Generative Fashion Designer on HF Spaces](https://huggingface.co/spaces/HashirAwaiz/generative-fashion-designer)**

---

## 🧠 Implemented Models

| # | Model | Key Techniques | Usage |
|---|-------|---------------|-----------|
| 1 | **β-VAE** | Residual encoder, KL annealing, reparameterization trick | Local |
| 2 | **DCGAN** | Spectral norm, self-attention, minibatch stddev | Local |
| 3 | **WGAN-GP** | Gradient penalty (λ=10), Wasserstein distance | Local |
| 4 | **Conditional GAN** | Projection discriminator, class embeddings | Local |
| 5 | **Latent DiT** | Diffusion Transformer in latent space | Local |
| 6 | **FLUX.2 Pro** | State-of-the-art cloud generation via Replicate | Cloud |

---

## 🚀 Quick Start

### 1. GPU Environment (RTX 4050 Optimized)
To train models locally on your GPU, use the dedicated Python 3.12 environment:
```powershell
# Activate and install (if not done)
.\venv_gpu\Scripts\activate
pip install -r requirements.txt

# Run training on GPU
.\venv_gpu\Scripts\python.exe scripts/train.py --model vae --epochs 50 --mixed-precision
```

### 2. Cloud Generation (FLUX.2 Pro)
Enable high-quality cloud generation by adding your Replicate token to a `.env` file:
```bash
REPLICATE_API_TOKEN=your_token_here
```

### 3. Training All Models
We have a convenient script to train the entire suite sequentially:
```powershell
.\venv_gpu\Scripts\python.exe scripts/train_all_models.py
```

---

## 🏗️ Architecture & Infrastructure

- **Web App**: Flask-based server served via Docker.
- **CI/CD**: Automated GitHub Actions pipeline that lints, tests, and deploys to Hugging Face.
- **Git LFS**: Large model weights (`.pt`) are tracked via Git LFS for seamless deployment.
- **Design**: Premium glassmorphic UI with real-time generation previews.

---

## ⚙️ Project Structure
```
generative-fashion-designer/
├── app/                         # Flask Backend & Frontend Assets
├── src/
│   ├── models/                  # VAE, GAN, WGAN, cGAN, DiT, Fusion
│   ├── training/                # Custom training engines
│   ├── inference/               # Unified generation & Cloud API
│   └── data/                    # Dataset & augmentation pipelines
├── .github/workflows/           # Automated CI/CD (Lint -> Test -> Deploy)
├── Dockerfile                   # Deployment container config
└── scripts/                     # CLI tools for training & evaluation
```

---

## 📈 Tech Stack
- **PyTorch 2.x** (CUDA 12.1)
- **Hugging Face Spaces** (Docker SDK)
- **Replicate SDK** (FLUX.2 Pro)
- **GitHub Actions** (CI/CD)
- **Git LFS** (Model Versioning)

---

## 📁 Evaluation Metrics
We implement manual versions of standard metrics to evaluate fashion generation:
- **FID** (Fréchet Inception Distance)
- **IS** (Inception Score)
- **SSIM** (Structural Similarity)

---

Created by **Hashir Awaiz** as part of an Advanced Deep Neural Networks study.
