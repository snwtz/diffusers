"""
python examples/multispectral/eval_multispectral_vae.py \
  --model_dir "C:/Users/NOcsPS-440g/Desktop/Zina/diffusers/examples/multispectral/Training_Split_18.06/test_run/final_model" \
  --val_file_list "C:/Users/NOcsPS-440g/Desktop/Zina/diffusers/examples/multispectral/Training_Split_18.06/val_files.txt" \
  --output_dir "C:/Users/NOcsPS-440g/Desktop/Zina/diffusers/examples/multispectral/Training_Split_18.06/eval_results" \
  --batch_size 1 \
  --num_samples 10
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

from diffusers import AutoencoderKLMultispectralAdapter
from vae_multispectral_dataloader import VAEMultispectralDataset
from torch.utils.data import DataLoader

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except ImportError:
    skimage_ssim = None
try:
    from pytorch_msssim import ssim as torch_ssim
except ImportError:
    torch_ssim = None

def plot_bands(original, reconstructed, outdir, idx):
    """Save side-by-side plots of all 5 bands for original and reconstructed images."""
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        axes[0, i].imshow(original[i], cmap='gray')
        axes[0, i].set_title(f'Orig Band {i+1}')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed[i], cmap='gray')
        axes[1, i].set_title(f'Recon Band {i+1}')
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'sample_{idx}_bands.png'))
    plt.close(fig)

def plot_spectral_signature(original, reconstructed, outdir, idx, n_points=5):
    """Plot spectral signatures (across bands) for a few random pixels."""
    os.makedirs(outdir, exist_ok=True)
    h, w = original.shape[1:]
    np.random.seed(42)
    # mask to visualize only leaf pixels (no all zero background)
    valid_mask = (original != 0).any(axis=0)  # shape: (H, W)
    ys, xs = np.where(valid_mask)
    if len(xs) < n_points:
        selected_indices = np.random.choice(len(xs), size=len(xs), replace=False)
    else:
        selected_indices = np.random.choice(len(xs), size=n_points, replace=False)
    points = [(ys[i], xs[i]) for i in selected_indices]
    fig, ax = plt.subplots()
    for y, x in points:
        orig_sig = original[:, y, x]
        recon_sig = reconstructed[:, y, x]
        ax.plot(range(5), orig_sig, 'o-', label=f'Orig ({y},{x})', alpha=0.7)
        ax.plot(range(5), recon_sig, 'x--', label=f'Recon ({y},{x})', alpha=0.7)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["B1","B2","B3","B4","B5"])
    ax.set_ylabel('Value')
    ax.set_title('Spectral Signatures')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'sample_{idx}_spectral_signature.png'))
    plt.close(fig)

def compute_sam(original, reconstructed, mask=None, eps=1e-8):
    """
    Compute Spectral Angle Mapper (SAM) between original and reconstructed images.
    
    Args:
        original: Original multispectral image (B, 5, H, W)
        reconstructed: Reconstructed multispectral image (B, 5, H, W)
        mask: Background mask (B, 1, H, W) where 1=leaf, 0=background
        eps: Small value to prevent division by zero
    
    Returns:
        Mean spectral angle in radians, computed only on leaf regions
    """
    # Apply mask if provided to focus only on leaf regions
    if mask is not None:
        # Expand mask to match channel dimension
        mask_expanded = mask.expand(-1, 5, -1, -1)  # (B, 5, H, W)
        original = original * mask_expanded
        reconstructed = reconstructed * mask_expanded
    
    # Flatten spatial dimensions
    orig_flat = original.reshape(original.shape[0], 5, -1)
    recon_flat = reconstructed.reshape(reconstructed.shape[0], 5, -1)
    
    # Compute dot product and norms
    dot = (orig_flat * recon_flat).sum(1)
    norm1 = np.linalg.norm(orig_flat, axis=1)
    norm2 = np.linalg.norm(recon_flat, axis=1)
    
    # Compute cosine similarity and clamp for stability
    cos = dot / (norm1 * norm2 + eps)
    cos = np.clip(cos, -1, 1)
    angle = np.arccos(cos)
    
    # If mask is provided, only average over valid (leaf) pixels
    if mask is not None:
        mask_flat = mask.reshape(mask.shape[0], -1)  # (B, H*W)
        valid_pixels = mask_flat.sum(axis=1) > 0  # (B,)
        if np.any(valid_pixels):
            return np.nanmean(angle[valid_pixels])
        else:
            return 0.0
    
    return np.nanmean(angle)

def compute_bandwise_mse(original, reconstructed, mask=None, eps=1e-8):
    """
    Compute per-band Mean Squared Error between original and reconstructed images.
    
    Args:
        original: Original multispectral image (B, 5, H, W)
        reconstructed: Reconstructed multispectral image (B, 5, H, W)
        mask: Background mask (B, 1, H, W) where 1=leaf, 0=background
        eps: Small value to prevent division by zero
    
    Returns:
        Per-band MSE, computed only on leaf regions if mask is provided
    """
    if mask is not None:
        # Apply mask to focus only on leaf regions
        mask_expanded = mask.expand(-1, 5, -1, -1)  # (B, 5, H, W)
        original = original * mask_expanded
        reconstructed = reconstructed * mask_expanded
        
        # Compute MSE only on valid (leaf) pixels
        squared_diff = (original - reconstructed) ** 2
        # Sum over spatial dimensions and divide by number of valid pixels per band
        mse_per_band = []
        for band in range(5):
            band_squared_diff = squared_diff[:, band]  # (B, H, W)
            band_mask = mask[:, 0]  # (B, H, W)
            # Sum squared differences over valid pixels, divide by count of valid pixels
            valid_pixels = band_mask.sum(dim=(1, 2))  # (B,)
            band_mse = (band_squared_diff * band_mask).sum(dim=(1, 2)) / (valid_pixels + eps)
            mse_per_band.append(band_mse.mean().item())
        return np.array(mse_per_band)
    else:
        # Fallback: compute MSE over entire image (includes background)
        mse = ((original - reconstructed) ** 2).mean(axis=(0,2,3))
        return mse

def compute_ssim(original, reconstructed, mask=None):
    """
    Compute Structural Similarity Index (SSIM) between original and reconstructed images.
    
    Args:
        original: Original multispectral image (B, 5, H, W)
        reconstructed: Reconstructed multispectral image (B, 5, H, W)
        mask: Background mask (B, 1, H, W) where 1=leaf, 0=background
    
    Returns:
        Per-band SSIM scores, computed only on leaf regions if mask is provided
    """
    ssim_scores = []
    if skimage_ssim is not None:
        # Use skimage: loop over batch and band
        for band in range(5):
            band_scores = []
            for i in range(original.shape[0]):
                orig_band = original[i, band]
                recon_band = reconstructed[i, band]
                
                if mask is not None:
                    sample_mask = mask[i, 0]  # (H, W)
                    # Only compute SSIM if there are valid leaf pixels
                    if sample_mask.sum() > 0:
                        # Apply mask to isolate leaf regions
                        masked_orig = orig_band * sample_mask
                        masked_recon = recon_band * sample_mask
                        
                        try:
                            # Try to use mask parameter if supported
                            score = skimage_ssim(masked_orig, masked_recon, data_range=2.0,
                                               mask=sample_mask.astype(bool))
                        except TypeError:
                            # Fallback: compute on entire image (less accurate)
                            score = skimage_ssim(orig_band, recon_band, data_range=2.0)
                    else:
                        score = 0.0  # No valid leaf regions
                else:
                    # No mask provided, compute on entire image
                    score = skimage_ssim(orig_band, recon_band, data_range=2.0)
                
                band_scores.append(score)
            ssim_scores.append(np.mean(band_scores))
    elif torch_ssim is not None:
        # Use pytorch_msssim: expects (N, C, H, W) and values in [0,1] or [-1,1]
        # We'll rescale from [-1,1] to [0,1] for compatibility
        orig = (original + 1) / 2
        recon = (reconstructed + 1) / 2
        
        if mask is not None:
            # Apply mask to focus only on leaf regions
            mask_expanded = mask.expand(-1, 5, -1, -1)
            orig = orig * mask_expanded
            recon = recon * mask_expanded
        
        for band in range(5):
            ssim_val = torch_ssim(
                torch.from_numpy(orig[:, band:band+1]),
                torch.from_numpy(recon[:, band:band+1]),
                data_range=1.0,
                size_average=True
            )
            ssim_scores.append(ssim_val.item())
    else:
        return None
    return ssim_scores

def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained multispectral VAE")
    parser.add_argument('--model_dir', type=str, required=True, help='Path to model directory (with config.json)')
    parser.add_argument('--val_file_list', type=str, required=True, help='Path to val/test file list')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='vae_eval_results')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoencoderKLMultispectralAdapter.from_pretrained(args.model_dir)
    model = model.to(device)
    model.eval()

    # CRITICAL: Load dataset with masks to enable leaf-only metric computation
    # This ensures we only evaluate reconstruction quality on leaf regions, not background
    dataset = VAEMultispectralDataset(
        file_list_path=args.val_file_list,
        resolution=512,
        use_cache=False,
        return_mask=True  # Enable mask loading for leaf-only evaluation
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_mse = []
    all_sam = []
    all_ssim = []
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, (batch, mask) in enumerate(tqdm(loader, desc='Evaluating')):
        # Handle both tuple and single tensor returns from dataloader
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        batch = batch.to(device)
        mask = mask.to(device)  # Background mask (1 for leaf, 0 for background)
        
        with torch.no_grad():
            recon, _ = model(batch)  # Model returns (reconstruction, losses), we only need reconstruction
        
        orig_np = batch.cpu().numpy()
        recon_np = recon.cpu().numpy()
        mask_np = mask.cpu().numpy()
        
        # Visualize
        if idx < args.num_samples:
            plot_bands(orig_np[0], recon_np[0], args.output_dir, idx)
            plot_spectral_signature(orig_np[0], recon_np[0], args.output_dir, idx)
        
        # CRITICAL: Apply mask to all metrics to evaluate only leaf regions
        # This prevents background pixels from artificially inflating metric scores
        all_mse.append(compute_bandwise_mse(orig_np, recon_np, mask_np))
        all_sam.append(compute_sam(orig_np, recon_np, mask_np))
        ssim_result = compute_ssim(orig_np, recon_np, mask_np)
        if ssim_result is not None:
            all_ssim.append(ssim_result)
    # Aggregate metrics
    all_mse = np.stack(all_mse)
    mean_mse = np.nanmean(all_mse, axis=0)
    mean_sam = np.nanmean(all_sam)
    if all_ssim:
        all_ssim = np.stack(all_ssim)
        mean_ssim = np.nanmean(all_ssim, axis=0)
    else:
        mean_ssim = None
    # Save metrics
    results = {
        'mean_mse_per_band': mean_mse.tolist(),
        'mean_sam': float(mean_sam),
        'mean_ssim_per_band': mean_ssim.tolist() if mean_ssim is not None else None
    }
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('Evaluation results:', results)
    if mean_ssim is not None:
        print("Mean SSIM per band:", mean_ssim)
    else:
        print("SSIM not computed (skimage or pytorch_msssim not installed).")

if __name__ == '__main__':
    main() 