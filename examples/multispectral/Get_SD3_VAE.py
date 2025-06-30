# "stabilityai/stable-diffusion-3-medium-diffusers" for diffusers pipeline.

from diffusers import DiffusionPipeline
import torch

# Download just the VAE by loading the full pipeline once
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16
)

# Save only the VAE (config and weights)
pipeline.vae.save_pretrained("./examples/multispectral/vae_only")