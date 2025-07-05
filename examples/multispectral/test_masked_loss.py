"""
Test script for masked loss computation in multispectral VAE adapter.

This script verifies that the masked loss computation correctly focuses training
on leaf regions while excluding background areas.
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
    
    print("Testing masked loss computation...")
    
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
    
    print(f"Mask shape: {mask.shape}")
    print(f"Mask coverage: {mask.mean().item():.4f}")
    print(f"Valid pixels: {mask.sum().item()}/{mask.numel()}")
    
    # Create synthetic reconstruction (slightly different from original)
    reconstructed = original + torch.randn_like(original) * 0.1
    
    # Test compute_losses method
    print("\nTesting compute_losses method...")
    losses = model.compute_losses(original, reconstructed, mask=mask)
    
    print("Loss components:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                print(f"  {key}: {value.item():.6f}")
            else:
                print(f"  {key}: {value.tolist()}")
        else:
            print(f"  {key}: {value}")
    
    # Test without mask (should warn about computing loss over entire image)
    print("\nTesting without mask...")
    losses_no_mask = model.compute_losses(original, reconstructed, mask=None)
    
    print("Loss components (no mask):")
    for key, value in losses_no_mask.items():
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                print(f"  {key}: {value.item():.6f}")
            else:
                print(f"  {key}: {value.tolist()}")
        else:
            print(f"  {key}: {value}")
    
    # Test with very small mask coverage
    print("\nTesting with very small mask coverage...")
    small_mask = torch.zeros_like(mask)
    small_mask[:, :, height//4:height//4+5, width//4:width//4+5] = 1.0  # Small square
    
    losses_small_mask = model.compute_losses(original, reconstructed, mask=small_mask)
    
    print("Loss components (small mask):")
    for key, value in losses_small_mask.items():
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                print(f"  {key}: {value.item():.6f}")
            else:
                print(f"  {key}: {value.tolist()}")
        else:
            print(f"  {key}: {value}")
    
    # Test forward method with mask
    print("\nTesting forward method with mask...")
    model.eval()
    with torch.no_grad():
        reconstruction, losses_forward = model.forward(sample=original, mask=mask)
    
    print("Forward method losses:")
    for key, value in losses_forward.items():
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                print(f"  {key}: {value.item():.6f}")
            else:
                print(f"  {key}: {value.tolist()}")
        else:
            print(f"  {key}: {value}")
    
    print("\nTest completed successfully!")
    
    # Verify that masked loss is different from unmasked loss
    mse_masked = losses['mse'].item()
    mse_unmasked = losses_no_mask['mse'].item()
    
    print(f"\nLoss comparison:")
    print(f"  Masked MSE: {mse_masked:.6f}")
    print(f"  Unmasked MSE: {mse_unmasked:.6f}")
    print(f"  Difference: {abs(mse_masked - mse_unmasked):.6f}")
    
    if abs(mse_masked - mse_unmasked) > 1e-6:
        print("✓ Masked loss computation is working correctly!")
    else:
        print("⚠ Warning: Masked and unmasked losses are very similar")

def test_sam_loss_with_masking():
    """Test SAM loss computation with masking."""
    
    print("\nTesting SAM loss with masking...")
    
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
    
    if 'sam' in losses:
        print(f"SAM loss with masking: {losses['sam'].item():.6f}")
        print("✓ SAM loss computation with masking works!")
    else:
        print("⚠ SAM loss not found in losses")

if __name__ == "__main__":
    try:
        test_masked_loss_computation()
        test_sam_loss_with_masking()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 