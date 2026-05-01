# Project Context: Generative Fashion Designer

## 1. Project Overview
[cite_start]A Generative Fashion Designer system to automate the rapid prototyping of traditional Pakistani textile patterns (Ajrak, Khaddar, Phulkari, Sindhi block prints, and Balochi embroidery)[cite: 9]. 
[cite_start]The system transforms a text prompt (and optionally a reference image) into a photorealistic, print-ready textile pattern[cite: 20].

## 2. Tech Stack
* **Frontend:** Next.js, Tailwind CSS (replaces Streamlit).
* **Backend:** FastAPI (Python 3.10).
* [cite_start]**Deep Learning Frameworks:** PyTorch 2.x, Hugging Face Diffusers (v0.27), Transformers (v4.x)[cite: 64].
* **Infrastructure:** Docker for containerization.
* **Hardware Profile:** Local development on RTX 4050 (6GB VRAM) - requires aggressive memory optimization for inference (`fp16`, attention slicing). Heavy LoRA training will be offloaded to Kaggle/Colab.

## 3. Deep Learning Architecture
[cite_start]The pipeline consists of three core components[cite: 20]:
1. [cite_start]**Prompt Interface:** Uses OpenAI's CLIP text encoder combined with LoRA adapters to map natural language to visual pattern semantics[cite: 21].
2. [cite_start]**Pattern Generator:** Fine-tuned Stable Diffusion v1.5 (or SDXL) to generate the base textile pattern[cite: 21].
3. [cite_start]**Style Transfer Module:** A custom PyTorch implementation of VGG-19 Neural Style Transfer to apply cultural visual style onto the generated base[cite: 21]. [cite_start]Optimization minimizes the weighted sum of content loss and style loss: $L\_total=\alpha\cdot L\_content+\beta\cdot L\_style$[cite: 39, 40].

## 4. System Pipeline (End-to-End)
* [cite_start]**Step 1:** User inputs text prompt + optional reference image via Next.js UI[cite: 51].
* **Step 2:** Request sent to FastAPI backend. [cite_start]CLIP encoder maps prompt to conditioning embedding[cite: 51].
* [cite_start]**Step 3:** Fine-tuned UNet iteratively denoises latent over $T=50$ DDIM steps[cite: 51].
* [cite_start]**Step 4:** VAE decodes latent to $512\times512$ RGB pattern image[cite: 51].
* [cite_start]**Step 5 (Optional):** VGG-19 NST blends reference swatch style onto the generated image[cite: 51].
* **Step 6:** Backend returns the image URL; Next.js allows user to download print-ready PNG + metadata.

## 5. Implementation Status
* **Phase 1 (Scaffolding):** Completed. Repository structure, FastAPI backend skeleton, and Next.js foundation established.
* **Phase 2 (UI & NST):** Completed. Next.js Dark Mode UI with form connectivity built. PyTorch VGG-19 Style Transfer class implemented with aggressive memory optimization (L-BFGS + memory clearing) for the 6GB VRAM constraint.