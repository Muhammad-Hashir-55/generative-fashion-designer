# ═══════════════════════════════════════════════════════════════
#  Generative Fashion Designer — Docker Image for HF Spaces
# ═══════════════════════════════════════════════════════════════

FROM python:3.10-slim

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Install CPU-only PyTorch (~1.5 GB instead of ~6 GB GPU build)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining project dependencies
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy project files
COPY . .

# Create required output directories with correct permissions
RUN mkdir -p outputs/checkpoints outputs/gallery outputs/generated outputs/evaluation data && \
    chown -R user:user /app

USER user

# Hugging Face Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app/server.py"]
