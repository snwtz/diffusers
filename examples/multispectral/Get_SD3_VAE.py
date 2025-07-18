# "stabilityai/stable-diffusion-3-medium-diffusers" for diffusers pipeline.

from diffusers import DiffusionPipeline
import torch
import os

# Download just the VAE by loading the full pipeline once
print("Loading SD3 pipeline...")
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16
)

# Save only the VAE (config and weights)
save_path = "./vae_only"
print(f"Saving VAE to: {os.path.abspath(save_path)}")
pipeline.vae.save_pretrained(save_path)
print(f"VAE saved successfully to: {save_path}")