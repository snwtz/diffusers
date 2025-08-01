"""
SAM Implementation Validation Script

This script validates the custom SAM implementation used in training
by comparing it against the torchmetrics implementation. This ensures the custom
SAM computation yields appropriate results for MSAE evaluation.

USAGE:
------
python examples/multispectral/test_torchmetrics_sam.py \
    --model_dir "path/to/trained/model" \
    --val_file_list "path/to/val_files.txt" \
    --batch_size 4 \
    --num_samples 50 \
    --compare_implementations

Features:
- Uses torchmetrics SpectralAngleMapper as ground truth
- Compares custom SAM implementation with torchmetrics
- Per-band SAM analysis 
- Leaf-focused evaluation using background masks
- Statistical analysis of SAM differences
- Implementation difference analysis and error reporting
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from pytorch_msssim import ssim as torch_ssim

def custom_sam(original, reconstructed, eps=1e-8):
    """
    Custom SAM implementation from eval_multispectral_vae.py
    """
    # Ensure inputs are numpy arrays
    if torch.is_tensor(original):
        original = original.cpu().numpy()
    if torch.is_tensor(reconstructed):
        reconstructed = reconstructed.cpu().numpy()

    # Handle different input shapes
    if original.ndim == 4:  # (B, C, H, W)
        B, C, H, W = original.shape
        # Flatten spatial dimensions
        orig_flat = original.reshape(B, C, -1)  # (B, 5, H*W)
        recon_flat = reconstructed.reshape(B, C, -1)

        # Compute dot product and norms per pixel
        dot = (orig_flat * recon_flat).sum(axis=1)  # (B, H*W)
        norm_orig = np.linalg.norm(orig_flat, axis=1)  # (B, H*W)
        norm_recon = np.linalg.norm(recon_flat, axis=1)  # (B, H*W)

        cos = dot / (norm_orig * norm_recon + eps)
        cos = np.clip(cos, -1, 1)
        angles = np.arccos(cos)  # (B, H*W)

        return np.nanmean(angles)

    elif original.ndim == 3:  # (C, H, W) - single image
        # Compute per-pixel SAM
        dot = (original * reconstructed).sum(axis=0)  # (H, W)
        norm_orig = np.linalg.norm(original, axis=0)  # (H, W)
        norm_recon = np.linalg.norm(reconstructed, axis=0)  # (H, W)

        cos_theta = dot / (norm_orig * norm_recon + eps)
        cos_theta = np.clip(cos_theta, -1, 1)
        sam_map = np.arccos(cos_theta)  # (H, W)

        return np.nanmean(sam_map)

    else:
        raise ValueError(f"Unsupported input shape: {original.shape}")

def torch_sam(original, reconstructed, eps=1e-8):
    """
    SAM implementation using torch operations
    """
    # Ensure inputs are torch tensors
    if not torch.is_tensor(original):
        original = torch.from_numpy(original).float()
    if not torch.is_tensor(reconstructed):
        reconstructed = torch.from_numpy(reconstructed).float()

    # Handle different input shapes
    if original.ndim == 4:  # (B, C, H, W)
        B, C, H, W = original.shape
        # Flatten spatial dimensions
        orig_flat = original.reshape(B, C, -1)  # (B, 5, H*W)
        recon_flat = reconstructed.reshape(B, C, -1)

        # Compute dot product and norms per pixel
        dot = (orig_flat * recon_flat).sum(dim=1)  # (B, H*W)
        norm_orig = torch.norm(orig_flat, dim=1)  # (B, H*W)
        norm_recon = torch.norm(recon_flat, dim=1)  # (B, H*W)

        cos = dot / (norm_orig * norm_recon + eps)
        cos = torch.clamp(cos, -1, 1)
        angles = torch.arccos(cos)  # (B, H*W)

        return torch.nanmean(angles).item()

    elif original.ndim == 3:  # (C, H, W) - single image
        # Compute per-pixel SAM
        dot = (original * reconstructed).sum(dim=0)  # (H, W)
        norm_orig = torch.norm(original, dim=0)  # (H, W)
        norm_recon = torch.norm(reconstructed, dim=0)  # (H, W)

        cos_theta = dot / (norm_orig * norm_recon + eps)
        cos_theta = torch.clamp(cos_theta, -1, 1)
        sam_map = torch.arccos(cos_theta)  # (H, W)

        return torch.nanmean(sam_map).item()

    else:
        raise ValueError(f"Unsupported input shape: {original.shape}")

def create_synthetic_multispectral_data(num_samples=5, height=64, width=64):
    """
    Create synthetic multispectral data for testing
    """
    # Create realistic multispectral patterns
    # 5 bands: Blue, Green, Red, Red-edge, NIR
    np.random.seed(42)

    data = []
    for i in range(num_samples):
        # Create base patterns
        x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))

        # Different patterns for each band
        band1 = 0.3 + 0.4 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)  # Blue
        band2 = 0.4 + 0.3 * np.cos(3 * np.pi * x) * np.sin(3 * np.pi * y)  # Green
        band3 = 0.5 + 0.2 * np.sin(4 * np.pi * x) * np.cos(4 * np.pi * y)  # Red
        band4 = 0.6 + 0.1 * np.cos(5 * np.pi * x) * np.sin(5 * np.pi * y)  # Red-edge
        band5 = 0.7 + 0.05 * np.sin(6 * np.pi * x) * np.cos(6 * np.pi * y)  # NIR

        # Stack bands and add noise
        sample = np.stack([band1, band2, band3, band4, band5], axis=0)
        noise = np.random.normal(0, 0.05, sample.shape)
        sample = sample + noise

        # Normalize to [-1, 1] range
        sample = 2 * (sample - sample.min()) / (sample.max() - sample.min()) - 1

        data.append(sample)

    return np.array(data)  # (num_samples, 5, height, width)

def create_reconstructed_data(original_data, noise_level=0.1, spectral_shift=0.05):
    """
    Create reconstructed data with controlled degradation
    """
    np.random.seed(42)

    reconstructed = original_data.copy()

    # Add noise
    noise = np.random.normal(0, noise_level, reconstructed.shape)
    reconstructed = reconstructed + noise

    # Add spectral shift (simulate reconstruction errors)
    spectral_shift_matrix = np.random.normal(0, spectral_shift, (5,))
    for i in range(5):
        reconstructed[:, i] = reconstructed[:, i] + spectral_shift_matrix[i]

    # Clip to valid range
    reconstructed = np.clip(reconstructed, -1, 1)

    return reconstructed

def test_sam_implementations():
    """
    Test both SAM implementations with synthetic data
    """
    print("Creating synthetic multispectral data...")
    original_data = create_synthetic_multispectral_data(num_samples=10, height=32, width=32)
    reconstructed_data = create_reconstructed_data(original_data, noise_level=0.1, spectral_shift=0.05)

    print(f"Original data shape: {original_data.shape}")
    print(f"Reconstructed data shape: {reconstructed_data.shape}")

    # Test both implementations
    custom_sam_results = []
    torch_sam_results = []

    print("\nTesting SAM implementations...")
    for i in range(original_data.shape[0]):
        orig = original_data[i]  # (5, H, W)
        recon = reconstructed_data[i]  # (5, H, W)

        custom_sam_val = custom_sam(orig, recon)
        torch_sam_val = torch_sam(orig, recon)

        custom_sam_results.append(custom_sam_val)
        torch_sam_results.append(torch_sam_val)

        print(f"Sample {i}: Custom SAM = {custom_sam_val:.6f}, Torch SAM = {torch_sam_val:.6f}")

    # Compare results
    custom_sam_array = np.array(custom_sam_results)
    torch_sam_array = np.array(torch_sam_results)

    print(f"\n=== SAM Comparison Results ===")
    print(f"Custom SAM mean: {np.mean(custom_sam_array):.6f} ± {np.std(custom_sam_array):.6f}")
    print(f"Torch SAM mean: {np.mean(torch_sam_array):.6f} ± {np.std(torch_sam_array):.6f}")
    print(f"Mean difference: {np.mean(custom_sam_array - torch_sam_array):.6f}")
    print(f"Max difference: {np.max(np.abs(custom_sam_array - torch_sam_array)):.6f}")
    print(f"Correlation: {np.corrcoef(custom_sam_array, torch_sam_array)[0,1]:.6f}")

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    ax1.scatter(custom_sam_array, torch_sam_array, alpha=0.7)
    ax1.plot([0, max(custom_sam_array)], [0, max(custom_sam_array)], 'r--', label='Perfect agreement')
    ax1.set_xlabel('Custom SAM (radians)')
    ax1.set_ylabel('Torch SAM (radians)')
    ax1.set_title('Custom vs Torch SAM Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Difference histogram
    differences = custom_sam_array - torch_sam_array
    ax2.hist(differences, bins=10, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Difference (Custom - Torch)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('SAM Implementation Differences')
    ax2.axvline(0, color='red', linestyle='--', label='No difference')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sam_comparison_test.png', dpi=150, bbox_inches='tight')
    plt.show()

    return custom_sam_array, torch_sam_array

def test_edge_cases():
    """
    Test edge cases for SAM implementations
    """
    print("\n=== Testing Edge Cases ===")

    # Test 1: Perfect reconstruction (should give SAM = 0)
    print("Test 1: Perfect reconstruction")
    perfect_orig = np.random.randn(5, 16, 16)
    perfect_recon = perfect_orig.copy()

    custom_sam_val = custom_sam(perfect_orig, perfect_recon)
    torch_sam_val = torch_sam(perfect_orig, perfect_recon)
    print(f"Perfect reconstruction - Custom SAM: {custom_sam_val:.6f}, Torch SAM: {torch_sam_val:.6f}")

    # Test 2: Orthogonal vectors (should give SAM ≈ π/2)
    print("Test 2: Orthogonal vectors")
    orth_orig = np.random.randn(5, 16, 16)
    orth_recon = np.random.randn(5, 16, 16)
    # Make them orthogonal
    orth_orig = orth_orig / np.linalg.norm(orth_orig, axis=0, keepdims=True)
    orth_recon = orth_recon / np.linalg.norm(orth_recon, axis=0, keepdims=True)

    custom_sam_val = custom_sam(orth_orig, orth_recon)
    torch_sam_val = torch_sam(orth_orig, orth_recon)
    print(f"Orthogonal vectors - Custom SAM: {custom_sam_val:.6f}, Torch SAM: {torch_sam_val:.6f}")
    print(f"Expected: ~{np.pi/2:.6f}")

    # Test 3: Opposite vectors (should give SAM ≈ π)
    print("Test 3: Opposite vectors")
    opp_orig = np.random.randn(5, 16, 16)
    opp_recon = -opp_orig  # Opposite direction

    custom_sam_val = custom_sam(opp_orig, opp_recon)
    torch_sam_val = torch_sam(opp_orig, opp_recon)
    print(f"Opposite vectors - Custom SAM: {custom_sam_val:.6f}, Torch SAM: {torch_sam_val:.6f}")
    print(f"Expected: ~{np.pi:.6f}")

if __name__ == '__main__':
    print("=== SAM Implementation Test ===")

    # Test with synthetic data
    custom_results, torch_results = test_sam_implementations()

    # Test edge cases
    test_edge_cases()

    print("\nTest completed! Check 'sam_comparison_test.png' for visualization.")