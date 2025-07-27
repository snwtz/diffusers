"""
Test script to compare custom SAM implementation with torchmetrics.image.SpectralAngleMapper.
Uses a VAE setup similar to eval_multispectral_vae.py.

Usage:
python examples/multispectral/sam_comparison_script.py \
  --model_dir "path/to/your/model" \
  --val_file_list "path/to/val_files.txt" \
  --batch_size 1 \
  --num_samples 5
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

from diffusers.models.autoencoders.autoencoder_kl_multispectral_adapter import AutoencoderKLMultispectralAdapter as AutoencoderKL
from torch.utils.data import DataLoader

# Try to import torchmetrics SpectralAngleMapper
try:
    from torchmetrics.image import SpectralAngleMapper
    TORCHMETRICS_SAM_AVAILABLE = True
    print("✓ torchmetrics.image.SpectralAngleMapper available")
except ImportError:
    TORCHMETRICS_SAM_AVAILABLE = False
    print("✗ torchmetrics.image.SpectralAngleMapper not available")

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

def torchmetrics_sam(original, reconstructed, eps=1e-8):
    """
    SAM implementation using torchmetrics.image.SpectralAngleMapper
    """
    if not TORCHMETRICS_SAM_AVAILABLE:
        return None
    
    # Ensure inputs are torch tensors
    if not torch.is_tensor(original):
        original = torch.from_numpy(original).float()
    if not torch.is_tensor(reconstructed):
        reconstructed = torch.from_numpy(reconstructed).float()
    
    # Handle different input shapes
    if original.ndim == 3:  # (C, H, W) - single image
        original = original.unsqueeze(0)  # Add batch dimension
    if reconstructed.ndim == 3:  # (C, H, W) - single image
        reconstructed = reconstructed.unsqueeze(0)  # Add batch dimension
    
    # Initialize SpectralAngleMapper
    sam_metric = SpectralAngleMapper(reduction='elementwise_mean')
    
    # Compute SAM (note: torchmetrics expects pred, target order)
    sam_value = sam_metric(reconstructed, original)
    
    return sam_value.item()

def compute_bandwise_sam(original, reconstructed, eps=1e-8):
    """
    Compute SAM per band (as implemented in the original script)
    """
    if torch.is_tensor(original):
        original = original.cpu().numpy()
    if torch.is_tensor(reconstructed):
        reconstructed = reconstructed.cpu().numpy()
    
    per_band_angles = []
    for band in range(5):
        orig_band = original[:, band]  # (B,H,W)
        recon_band = reconstructed[:, band]  # (B,H,W)
        # Flatten spatial dims
        orig_b_flat = orig_band.reshape(orig_band.shape[0], -1)
        recon_b_flat = recon_band.reshape(recon_band.shape[0], -1)
        dot_b = orig_b_flat * recon_b_flat  # (B,H*W)
        norm_orig_b = np.abs(orig_b_flat)
        norm_recon_b = np.abs(recon_b_flat)
        cos_b = dot_b / (norm_orig_b * norm_recon_b + eps)
        cos_b = np.clip(cos_b, -1, 1)
        angles_b = np.arccos(cos_b)  # (B,H*W)
        # Mask out zero norm pixels to avoid invalid angles
        valid_mask_b = (norm_orig_b > eps) & (norm_recon_b > eps)
        if np.any(valid_mask_b):
            mean_angle_b = np.nanmean(angles_b[valid_mask_b])
        else:
            mean_angle_b = 0.0
        per_band_angles.append(mean_angle_b)
    
    return np.array(per_band_angles)

def plot_sam_comparison(custom_sam_values, torchmetrics_sam_values, output_dir):
    """Plot comparison between custom and torchmetrics SAM implementations"""
    os.makedirs(output_dir, exist_ok=True)
    
    if torchmetrics_sam_values is None:
        print("TorchMetrics SAM not available, skipping comparison plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    ax1.scatter(custom_sam_values, torchmetrics_sam_values, alpha=0.6)
    ax1.plot([0, max(custom_sam_values)], [0, max(custom_sam_values)], 'r--', label='Perfect agreement')
    ax1.set_xlabel('Custom SAM (radians)')
    ax1.set_ylabel('TorchMetrics SAM (radians)')
    ax1.set_title('Custom vs TorchMetrics SAM Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Difference histogram
    differences = np.array(custom_sam_values) - np.array(torchmetrics_sam_values)
    ax2.hist(differences, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Difference (Custom - TorchMetrics)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('SAM Implementation Differences')
    ax2.axvline(0, color='red', linestyle='--', label='No difference')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sam_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"\nSAM Comparison Statistics:")
    print(f"Custom SAM mean: {np.mean(custom_sam_values):.6f} ± {np.std(custom_sam_values):.6f}")
    print(f"TorchMetrics SAM mean: {np.mean(torchmetrics_sam_values):.6f} ± {np.std(torchmetrics_sam_values):.6f}")
    print(f"Mean difference: {np.mean(differences):.6f}")
    print(f"Max difference: {np.max(np.abs(differences)):.6f}")
    print(f"Correlation: {np.corrcoef(custom_sam_values, torchmetrics_sam_values)[0,1]:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Compare custom SAM with torchmetrics.image.SpectralAngleMapper implementation")
    parser.add_argument('--model_dir', type=str, required=True, help='Path to model directory')
    parser.add_argument('--val_file_list', type=str, required=True, help='Path to validation file list (text file containing paths to TIFF files)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to process')
    parser.add_argument('--output_dir', type=str, default='sam_comparison_results', help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model (similar to eval script)
    model = AutoencoderKL.from_pretrained(
        args.model_dir,
        adapter_in_channels=5,
        adapter_out_channels=5,
        backbone_in_channels=3,
        backbone_out_channels=3,
        latent_channels=16,
        adapter_placement="both",
        use_spectral_attention=True,
        use_sam_loss=True,
        use_saturation_penalty=True,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
    )
    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully")

    # Load dataset
    try:
        dataset = VAEMultispectralDataset(
            file_list_path=args.val_file_list,
            resolution=512,
            use_cache=False,
            return_mask=True
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        print(f"✓ Dataset loaded with {len(dataset)} samples")
    except FileNotFoundError as e:
        print(f"❌ Error loading dataset: {e}")
        print(f"   The --val_file_list should point to a text file containing paths to TIFF files.")
        print(f"   Example: --val_file_list 'examples/multispectral/Training_Split_18.06/val_files.txt'")
        print(f"   Not a directory path.")
        return
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # Results storage
    custom_sam_results = []
    torchmetrics_sam_results = []
    bandwise_sam_results = []
    sample_info = []

    os.makedirs(args.output_dir, exist_ok=True)

    # Process samples
    for idx, (batch, mask) in enumerate(tqdm(loader, desc='Processing samples')):
        if idx >= args.num_samples:
            break
            
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        
        batch = batch.to(device)
        mask = mask.to(device)
        
        with torch.no_grad():
            recon = model(batch)
            if hasattr(recon, "sample"):
                decoded_tensor = recon.sample
            else:
                decoded_tensor = recon

        # Convert to numpy for processing
        orig_np = batch.cpu().numpy()
        recon_np = decoded_tensor.cpu().numpy()
        
        # Compute SAM using both implementations
        custom_sam_val = custom_sam(orig_np, recon_np)
        torchmetrics_sam_val = torchmetrics_sam(orig_np, recon_np)
        bandwise_sam = compute_bandwise_sam(orig_np, recon_np)
        
        # Store results
        custom_sam_results.append(custom_sam_val)
        if torchmetrics_sam_val is not None:
            torchmetrics_sam_results.append(torchmetrics_sam_val)
        bandwise_sam_results.append(bandwise_sam)
        
        # Store sample info
        sample_info.append({
            'sample_idx': idx,
            'custom_sam': custom_sam_val,
            'torchmetrics_sam': torchmetrics_sam_val,
            'bandwise_sam': bandwise_sam.tolist(),
            'original_range': [float(orig_np.min()), float(orig_np.max())],
            'reconstructed_range': [float(recon_np.min()), float(recon_np.max())]
        })
        
        print(f"Sample {idx}: Custom SAM = {custom_sam_val:.6f}, TorchMetrics SAM = {torchmetrics_sam_val:.6f if torchmetrics_sam_val is not None else 'N/A'}")

    # Create comparison plot
    if torchmetrics_sam_results:
        plot_sam_comparison(custom_sam_results, torchmetrics_sam_results, args.output_dir)
    
    # Save detailed results
    results = {
        'summary': {
            'num_samples': len(custom_sam_results),
            'custom_sam_mean': float(np.mean(custom_sam_results)),
            'custom_sam_std': float(np.std(custom_sam_results)),
            'torchmetrics_sam_mean': float(np.mean(torchmetrics_sam_results)) if torchmetrics_sam_results else None,
            'torchmetrics_sam_std': float(np.std(torchmetrics_sam_results)) if torchmetrics_sam_results else None,
        },
        'bandwise_summary': {
            'mean_per_band': np.mean(bandwise_sam_results, axis=0).tolist(),
            'std_per_band': np.std(bandwise_sam_results, axis=0).tolist(),
        },
        'sample_details': sample_info
    }
    
    with open(os.path.join(args.output_dir, 'sam_comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n=== SAM Comparison Summary ===")
    print(f"Processed {len(custom_sam_results)} samples")
    print(f"Custom SAM: {results['summary']['custom_sam_mean']:.6f} ± {results['summary']['custom_sam_std']:.6f}")
    if torchmetrics_sam_results:
        print(f"TorchMetrics SAM: {results['summary']['torchmetrics_sam_mean']:.6f} ± {results['summary']['torchmetrics_sam_std']:.6f}")
    print(f"Band-wise SAM: {[f'{v:.6f}' for v in results['bandwise_summary']['mean_per_band']]}")
    print(f"Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main() 