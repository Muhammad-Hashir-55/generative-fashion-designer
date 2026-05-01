"""
Model architectures for generative fashion design.
Includes VAE, DCGAN, WGAN-GP, Conditional GAN, Neural Style Transfer,
and the CVAE-GAN Hybrid Fusion model.
"""

from src.models.vae import VAE
from src.models.dcgan import DCGenerator, DCDiscriminator
from src.models.wgan_gp import WGANGenerator, WGANCritic, compute_gradient_penalty
from src.models.conditional_gan import ConditionalGenerator, ProjectionDiscriminator
from src.models.style_transfer import NeuralStyleTransfer
from src.models.fusion_generator import CVAEGANFusion

__all__ = [
    "VAE",
    "DCGenerator",
    "DCDiscriminator",
    "WGANGenerator",
    "WGANCritic",
    "compute_gradient_penalty",
    "ConditionalGenerator",
    "ProjectionDiscriminator",
    "NeuralStyleTransfer",
    "CVAEGANFusion",
]
