"""
Test script for masked loss computation in multispectral VAE adapter.

This script verifies that the masked loss computation correctly focuses training
on leaf regions while excluding background areas.

Purpose in detail:
1. Verify pure leaf-focused training - ensuring the model learns only from biologically relevant leaf regions while excluding background areas
2. Test masked loss computation - validating that the loss functions correctly focus on leaf regions using binary masks
3. Validate SAM (Spectral Angle Mapper) loss - ensuring spectral fidelity is maintained when using masks
4. Provide debugging and monitoring tools - helping track mask coverage and loss behavior during training

Use cases:
	•	Changing loss functions
	•	Suspecting training problems
	•	Comparing masked/unmasked behavior
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the path to import the VAE adapter
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from diffusers.models.autoencoders.autoencoder_kl_multispectral_adapter import (
    AutoencoderKLMultispectralAdapter,
    compute_sam_loss
)

def test_masked_loss_computation():
    """Test the masked loss computation with synthetic data."""
    
    # Create a simple model for testing
    model = AutoencoderKLMultispectralAdapter(
        pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers",
        adapter_placement="both",
        use_spectral_attention=True,
        use_sam_loss=True
    )
    
    # Create synthetic test data
    batch_size = 2
    height, width = 64, 64
    num_channels = 5
    
    # Create synthetic multispectral images
    original = torch.randn(batch_size, num_channels, height, width) * 0.5  # Range roughly [-1, 1]
    
    # Create synthetic masks (1 for leaf, 0 for background)
    # Create a circular mask to simulate a leaf
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    center_y, center_x = height // 2, width // 2
    radius = min(height, width) // 3
    
    # Create circular mask
    mask = ((y - center_y) ** 2 + (x - center_x) ** 2) < radius ** 2
    mask = mask.float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    mask = mask.expand(batch_size, 1, height, width)  # Expand to batch size
    
    # Create synthetic reconstruction (slightly different from original)
    reconstructed = original + torch.randn_like(original) * 0.1
    
    # Test compute_losses method
    losses = model.compute_losses(original, reconstructed, mask=mask)
    
    # Test without mask (should warn about computing loss over entire image)
    losses_no_mask = model.compute_losses(original, reconstructed, mask=None)
    
    # Test with very small mask coverage
    small_mask = torch.zeros_like(mask)
    small_mask[:, :, height//4:height//4+5, width//4:width//4+5] = 1.0  # Small square
    
    losses_small_mask = model.compute_losses(original, reconstructed, mask=small_mask)
    
    # Test forward method with mask
    model.eval()
    with torch.no_grad():
        reconstruction, losses_forward = model.forward(sample=original, mask=mask)
    
    # Assert that forward method outputs have expected types and shapes
    assert isinstance(reconstruction, torch.Tensor)
    assert reconstruction.shape == original.shape
    assert isinstance(losses_forward, dict)
    for key, value in losses_forward.items():
        assert isinstance(value, torch.Tensor) or isinstance(value, float)
    
    # Verify that masked loss is different from unmasked loss
    mse_masked = losses['mse'].item()
    mse_unmasked = losses_no_mask['mse'].item()
    
    assert abs(mse_masked - mse_unmasked) > 1e-6, "Masked and unmasked MSE losses are too similar"
    
    # Placeholder for loading real .tiff + mask pair in future
    # e.g., real_image, real_mask = load_real_tiff_and_mask(...)
    # This would be used to test masked loss on real data

def test_sam_loss_with_masking():
    """Test SAM loss computation with masking."""
    
    # Create synthetic data
    batch_size = 2
    height, width = 32, 32
    num_channels = 5
    
    # Create synthetic multispectral images
    original = torch.randn(batch_size, num_channels, height, width) * 0.5
    
    # Create mask with some valid regions
    mask = torch.zeros(batch_size, 1, height, width)
    mask[:, :, height//4:3*height//4, width//4:3*width//4] = 1.0  # Center region
    
    # Create reconstruction
    reconstructed = original + torch.randn_like(original) * 0.1
    
    # Test SAM loss computation
    model = AutoencoderKLMultispectralAdapter(
        pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers",
        adapter_placement="both",
        use_spectral_attention=True,
        use_sam_loss=True
    )
    
    losses = model.compute_losses(original, reconstructed, mask=mask)
    
    assert 'sam' in losses, "SAM loss not found in losses"

if __name__ == "__main__":
    try:
        test_masked_loss_computation()
        test_sam_loss_with_masking()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e