"""
python examples/multispectral/eval_multispectral_vae.py \
  --model_dir "C:/Users/NOcsPS-440g/Desktop/Zina/diffusers/examples/multispectral/Training_Split_18.06/test_run/final_model" \
  --val_file_list "C:/Users/NOcsPS-440g/Desktop/Zina/diffusers/examples/multispectral/Training_Split_18.06/val_files.txt" \
  --output_dir "C:/Users/NOcsPS-440g/Desktop/Zina/diffusers/examples/multispectral/Training_Split_18.06/eval_results" \
  --batch_size 1 \
  --num_samples 10

Note: This evaluation script now properly handles [-1, 1] normalized data throughout the pipeline.
The dataloader normalizes to [-1, 1], the model expects [-1, 1], and visualization converts to [0, 1] for display.

AutoencoderKL is not imported or defined anywhere. All evaluated configs are from adapters only 
-> inference checks MS VAE model.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

from diffusers.models.autoencoders.autoencoder_kl_multispectral_adapter import AutoencoderKLMultispectralAdapter as AutoencoderKL
# from diffusers import AutoencoderKLMultispectralAdapter
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

def plot_pseudo_rgb(original, reconstructed, outdir, idx):
    """
    Create pseudo-RGB visualization of multispectral images.
    
    Maps the 5 spectral bands to RGB channels for human visualization:
    - Band 1 (474.73nm - Blue) -> Blue channel
    - Band 2 (538.71nm - Green) -> Green channel  
    - Band 3 (650.665nm - Red) -> Red channel
    - Band 4 (730.635nm - Red-edge) -> Additional red contribution
    - Band 5 (850.59nm - NIR) -> Additional green contribution
    
    This mapping follows the biological relevance of each band for plant health analysis.
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Create pseudo-RGB mapping based on spectral characteristics
    
    def create_pseudo_rgb(bands):
        # Normalize from [-1, 1] to [0, 1] for RGB visualization
        bands_norm = (bands + 1) / 2
        
        # Create RGB channels with spectral mapping
        r = bands_norm[2] * 0.7 + bands_norm[3] * 0.3  # Red + Red-edge
        g = bands_norm[1] * 0.6 + bands_norm[4] * 0.4  # Green + NIR
        b = bands_norm[0]  # Blue
        
        # Stack channels and ensure proper range
        rgb = np.stack([r, g, b], axis=0)
        rgb = np.clip(rgb, 0, 1)
        return rgb
    
    # Create pseudo-RGB for original and reconstructed
    orig_rgb = create_pseudo_rgb(original)
    recon_rgb = create_pseudo_rgb(reconstructed)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original pseudo-RGB
    axes[0].imshow(orig_rgb.transpose(1, 2, 0))
    axes[0].set_title('Original Pseudo-RGB\n(B1→B, B2→G, B3→R, B4→R+, B5→G+)')
    axes[0].axis('off')
    
    # Reconstructed pseudo-RGB
    axes[1].imshow(recon_rgb.transpose(1, 2, 0))
    axes[1].set_title('Reconstructed Pseudo-RGB\n(B1→B, B2→G, B3→R, B4→R+, B5→G+)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'sample_{idx}_pseudo_rgb.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Also save individual RGB channels for detailed analysis
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original channels
    axes[0, 0].imshow(orig_rgb[0], cmap='Blues', vmin=0, vmax=1)
    axes[0, 0].set_title('Original - Blue (B1: 474.73nm)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(orig_rgb[1], cmap='Greens', vmin=0, vmax=1)
    axes[0, 1].set_title('Original - Green (B2: 538.71nm + B5: 850.59nm)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(orig_rgb[2], cmap='Reds', vmin=0, vmax=1)
    axes[0, 2].set_title('Original - Red (B3: 650.665nm + B4: 730.635nm)')
    axes[0, 2].axis('off')
    
    # Reconstructed channels
    axes[1, 0].imshow(recon_rgb[0], cmap='Blues', vmin=0, vmax=1)
    axes[1, 0].set_title('Reconstructed - Blue (B1: 474.73nm)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(recon_rgb[1], cmap='Greens', vmin=0, vmax=1)
    axes[1, 1].set_title('Reconstructed - Green (B2: 538.71nm + B5: 850.59nm)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(recon_rgb[2], cmap='Reds', vmin=0, vmax=1)
    axes[1, 2].set_title('Reconstructed - Red (B3: 650.665nm + B4: 730.635nm)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'sample_{idx}_pseudo_rgb_channels.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_bands(original, reconstructed, outdir, idx):
    """Save side-by-side plots of all 5 bands for original and reconstructed images."""
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        # Data is in [-1, 1] range, normalize to [0, 1] for visualization
        orig_norm = (original[i] + 1) / 2  # [-1, 1] -> [0, 1]
        recon_norm = (reconstructed[i] + 1) / 2  # [-1, 1] -> [0, 1]
        axes[0, i].imshow(orig_norm, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Orig Band {i+1}')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon_norm, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Recon Band {i+1}')
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'sample_{idx}_bands.png'))
    plt.close(fig)

def plot_error_maps(original, reconstructed, mask, outdir, idx, points=None):
    """Generate and save per-pixel error heatmaps for MSE and SAM.
    Optionally overlays red markers for provided (y, x) coordinates."""
    os.makedirs(outdir, exist_ok=True)
    mse_map = ((original - reconstructed) ** 2).mean(axis=0)  # (H, W)
    # Normalize for plotting
    mse_map_norm = (mse_map - mse_map.min()) / (mse_map.max() - mse_map.min() + 1e-8)

    # Compute SAM map per pixel
    dot = (original * reconstructed).sum(axis=0)
    norm_orig = np.linalg.norm(original, axis=0)
    norm_recon = np.linalg.norm(reconstructed, axis=0)
    cos_theta = dot / (norm_orig * norm_recon + 1e-8)
    cos_theta = np.clip(cos_theta, -1, 1)
    sam_map = np.arccos(cos_theta)
    sam_map_norm = sam_map / np.pi  # normalize to [0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(mse_map_norm, ax=axes[0], cmap='viridis')
    axes[0].set_title("Per-pixel MSE (normalized)")
    axes[0].axis("off")

    sns.heatmap(sam_map_norm, ax=axes[1], cmap='magma')
    axes[1].set_title("Per-pixel SAM (normalized)")
    axes[1].axis("off")

    # Overlay red markers for selected points if provided, with numerical labels
    if points is not None:
        for idx_p, (y, x) in enumerate(points):
            axes[0].plot(x, y, 'ro', markersize=3)
            axes[1].plot(x, y, 'ro', markersize=3)
            axes[0].text(x + 2, y, str(idx_p + 1), color='red', fontsize=8)
            axes[1].text(x + 2, y, str(idx_p + 1), color='red', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'sample_{idx}_error_maps.png'))
    plt.close(fig)

def plot_spectral_signature(original, reconstructed, mask, outdir, idx, n_points=5):
    """Plot spectral signatures (across bands) for a few random pixels.
    
    Note: Only pixels within valid (leaf) regions are sampled to reflect leaf fidelity.
    Returns the list of selected (y, x) coordinates.
    """
    os.makedirs(outdir, exist_ok=True)
    h, w = original.shape[1:]
    np.random.seed(42)
    # mask == 1 marks leaf (foreground); restrict sampling to leaf pixels only
    valid_mask = mask[0, 0] > 0  # Corrected shape handling: (1, H, W) -> (H, W)
    ys, xs = np.where(valid_mask)
    if len(xs) < n_points:
        selected_indices = np.random.choice(len(xs), size=len(xs), replace=False)
    else:
        selected_indices = np.random.choice(len(xs), size=n_points, replace=False)
    points = [(ys[i], xs[i]) for i in selected_indices]
    fig, ax = plt.subplots()
    for idx_p, (y, x) in enumerate(points):
        orig_sig = original[:, y, x]
        recon_sig = reconstructed[:, y, x]
        # Data is in [-1, 1] range, normalize to [0, 1] for plotting
        orig_sig_norm = (orig_sig + 1) / 2  # [-1, 1] -> [0, 1]
        recon_sig_norm = (recon_sig + 1) / 2  # [-1, 1] -> [0, 1]
        ax.plot(range(5), orig_sig_norm, 'o-', label=f'Pixel {idx_p+1} Orig', alpha=0.7)
        ax.plot(range(5), recon_sig_norm, 'x--', label=f'Pixel {idx_p+1} Recon', alpha=0.7)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["B1","B2","B3","B4","B5"])
    ax.set_ylabel('Value')
    ax.set_title('Spectral Signatures')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'sample_{idx}_spectral_signature.png'))
    # Return the points used for plotting
    return points
    plt.close(fig)

# New function to aggregate spectral errors
def aggregate_spectral_errors(all_errors, outdir):
    """
    Plots the mean absolute difference between the original and reconstructed pixel values 
    across all sampled spectral signatures, computed per spectral band.
    Args:
        all_errors: List of arrays (N_points, 5) containing abs(orig - recon) values.
        outdir: Path to save the summary plot.
    """
    if not all_errors:
        return
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    all_errors_stacked = np.vstack(all_errors)  # shape: (total_points, 5)
    mean_abs_error = np.mean(all_errors_stacked, axis=0)
    std_abs_error = np.std(all_errors_stacked, axis=0)

    plt.figure()
    plt.bar(range(5), mean_abs_error, yerr=std_abs_error, capsize=5)
    plt.xticks(range(5), ["B1", "B2", "B3", "B4", "B5"])
    plt.ylabel("Mean Absolute Error")
    plt.title("Band-wise Spectral Signature Reconstruction Error (based on sample pixels)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "spectral_signature_error_summary.png"))
    plt.close()

def compute_sam(original, reconstructed, mask=None, eps=1e-8):
    """
    Compute Spectral Angle Mapper (SAM) between original and reconstructed images.
    Computes three sets of metrics if mask is provided:
    - Foreground (mask==1): focuses on leaf fidelity
    - Background (mask==0): provides context on background reconstruction
    - Overall (no mask): global metric over entire image
    
    Additionally, computes per-band SAM by averaging spectral angles per band.
    NOTE: SAM is angular, so band-wise breakdown is a bit unconventional, 
    but here it's computed sensibly: angle between original and reconstructed 
    vector using a one-hot band vector
    
    Args:
        original: Original multispectral image (B, 5, H, W)
        reconstructed: Reconstructed multispectral image (B, 5, H, W)
        mask: Background mask (B, 1, H, W) where 1=leaf, 0=background
        eps: Small value to prevent division by zero
    
    Returns:
        Dictionary with keys 'foreground', 'background', and 'overall', each containing:
            - 'mean_angle': Mean spectral angle in radians
            - 'per_band_angle': Per-band mean spectral angle in radians (length 5)
    """
    def sam_per_mask(m):
        # m shape: (B,1,H,W) or None
        if m is not None:
            mask_expanded = np.repeat(m, 5, axis=1)  # (B,5,H,W)
            orig_masked = original * mask_expanded
            recon_masked = reconstructed * mask_expanded
        else:
            orig_masked = original
            recon_masked = reconstructed
        
        B, C, H, W = orig_masked.shape
        # Flatten spatial dimensions
        orig_flat = orig_masked.reshape(B, C, -1)  # (B,5,H*W)
        recon_flat = recon_masked.reshape(B, C, -1)
        
        # Compute dot product and norms per pixel
        dot = (orig_flat * recon_flat).sum(axis=1)  # (B, H*W)
        norm_orig = np.linalg.norm(orig_flat, axis=1)  # (B, H*W)
        norm_recon = np.linalg.norm(recon_flat, axis=1)  # (B, H*W)
        
        cos = dot / (norm_orig * norm_recon + eps)
        cos = np.clip(cos, -1, 1)
        angles = np.arccos(cos)  # (B, H*W)
        
        # Per-band SAM: compute angle per band by considering each band's vector as 1D
        # Here, we compute SAM per band by comparing original and reconstructed values per band
        # Since each band is scalar, spectral angle reduces to 0 if values have same sign and magnitude
        # Instead, we compute mean absolute difference normalized by magnitude for each band
        per_band_angles = []
        for band in range(5):
            orig_band = orig_masked[:, band]  # (B,H,W)
            recon_band = recon_masked[:, band]  # (B,H,W)
            # Flatten spatial dims
            orig_b_flat = orig_band.reshape(B, -1)
            recon_b_flat = recon_band.reshape(B, -1)
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
        per_band_angles = np.array(per_band_angles)
        
        if m is not None:
            mask_flat = m.reshape(B, -1)  # (B,H*W)
            valid_pixels = mask_flat.sum(axis=1) > 0  # (B,)
            if np.any(valid_pixels):
                mean_angle = np.nanmean(angles[valid_pixels])
            else:
                mean_angle = 0.0
        else:
            mean_angle = np.nanmean(angles)
        
        return {'mean_angle': mean_angle, 'per_band_angle': per_band_angles}
    
    results = {}
    # Foreground metrics (mask==1)
    if mask is not None:
        results['foreground'] = sam_per_mask(mask)
        # Background metrics (mask==0)
        background_mask = 1 - mask
        results['background'] = sam_per_mask(background_mask)
    else:
        results['foreground'] = None
        results['background'] = None
    # Overall metrics (no mask)
    results['overall'] = sam_per_mask(None)
    # Add per-band angles for each region
    results['per_band_angle_foreground'] = results['foreground']['per_band_angle'].tolist() if results['foreground'] is not None else None
    results['per_band_angle_background'] = results['background']['per_band_angle'].tolist() if results['background'] is not None else None
    results['per_band_angle_overall'] = results['overall']['per_band_angle'].tolist()
    return results

def compute_bandwise_mse(original, reconstructed, mask=None, eps=1e-8):
    """
    Compute per-band Mean Squared Error (MSE) between original and reconstructed images.
    Computes three sets of metrics if mask is provided:
    - Foreground (mask==1): focuses on leaf fidelity
    - Background (mask==0): provides context on background reconstruction
    - Overall (no mask): global metric over entire image
    
    Args:
        original: Original multispectral image (B, 5, H, W)
        reconstructed: Reconstructed multispectral image (B, 5, H, W)
        mask: Background mask (B, 1, H, W) where 1=leaf, 0=background
        eps: Small value to prevent division by zero
    
    Returns:
        Dictionary with keys 'foreground', 'background', and 'overall', each containing:
            - Per-band MSE (length 5)
    """
    def mse_per_mask(m):
        if m is not None:
            mask_expanded = np.repeat(m, 5, axis=1)  # (B, 5, H, W)
            orig_masked = original * mask_expanded
            recon_masked = reconstructed * mask_expanded
            
            squared_diff = (orig_masked - recon_masked) ** 2
            mse_per_band = []
            for band in range(5):
                band_squared_diff = squared_diff[:, band]  # (B, H, W)
                band_mask = m[:, 0]  # (B, H, W)
                valid_pixels = band_mask.sum(axis=(1, 2))  # (B,)
                band_mse = (band_squared_diff * band_mask).sum(axis=(1, 2)) / (valid_pixels + eps)
                mse_per_band.append(band_mse.mean())
            return np.array(mse_per_band)
        else:
            mse = ((original - reconstructed) ** 2).mean(axis=(0,2,3))
            return mse
    results = {}
    if mask is not None:
        results['foreground'] = mse_per_mask(mask)
        background_mask = 1 - mask
        results['background'] = mse_per_mask(background_mask)
    else:
        results['foreground'] = None
        results['background'] = None
    results['overall'] = mse_per_mask(None)
    return results

def compute_ssim(original, reconstructed, mask=None):
    """
    Compute Structural Similarity Index (SSIM) between original and reconstructed images.
    Computes three sets of metrics if mask is provided:
    - Foreground (mask==1): focuses on leaf fidelity
    - Background (mask==0): provides context on background reconstruction
    - Overall (no mask): global metric over entire image
    
    Args:
        original: Original multispectral image (B, 5, H, W)
        reconstructed: Reconstructed multispectral image (B, 5, H, W)
        mask: Background mask (B, 1, H, W) where 1=leaf, 0=background
    
    Returns:
        Dictionary with keys 'foreground', 'background', and 'overall', each containing:
            - Per-band SSIM scores (length 5)
    """
    def ssim_per_mask(m):
        ssim_scores = []
        if skimage_ssim is not None:
            for band in range(5):
                band_scores = []
                for i in range(original.shape[0]):
                    orig_band = original[i, band]
                    recon_band = reconstructed[i, band]
                    
                    if m is not None:
                        sample_mask = m[i, 0]  # (H, W)
                        if sample_mask.sum() > 0:
                            masked_orig = orig_band * sample_mask
                            masked_recon = recon_band * sample_mask
                            try:
                                score = skimage_ssim(masked_orig, masked_recon, data_range=2.0,  # [-1, 1] range = 2.0
                                                   mask=sample_mask.astype(bool))
                            except TypeError:
                                score = skimage_ssim(orig_band, recon_band, data_range=2.0)  # [-1, 1] range = 2.0
                        else:
                            score = 0.0
                    else:
                        score = skimage_ssim(orig_band, recon_band, data_range=2.0)  # [-1, 1] range = 2.0
                    band_scores.append(score)
                ssim_scores.append(np.mean(band_scores))
        elif torch_ssim is not None:
            # model now outputs values in [0, 1] in v3
            orig = original
            recon = reconstructed
            if m is not None:
                mask_expanded = np.repeat(m, 5, axis=1)
                orig = orig * mask_expanded
                recon = recon * mask_expanded
            for band in range(5):
                ssim_val = torch_ssim(
                    torch.from_numpy(orig[:, band:band+1]),
                    torch.from_numpy(recon[:, band:band+1]),
                    data_range=2.0,  # [-1, 1] range = 2.0
                    size_average=True
                )
                ssim_scores.append(ssim_val.item())
        else:
            return None
        return ssim_scores
    
    results = {}
    if mask is not None:
        results['foreground'] = ssim_per_mask(mask)
        background_mask = 1 - mask
        results['background'] = ssim_per_mask(background_mask)
    else:
        results['foreground'] = None
        results['background'] = None
    results['overall'] = ssim_per_mask(None)
    return results

def check_normalization_consistency(original, reconstructed, outdir, idx):
    """Check for potential light/dark inversions and normalization issues."""
    os.makedirs(outdir, exist_ok=True)
    
    # Check data ranges
    orig_min, orig_max = original.min(), original.max()
    recon_min, recon_max = reconstructed.min(), reconstructed.max()
    
    # Check for potential inversions by comparing mean values per band
    orig_means = original.mean(axis=(1, 2))  # (5,)
    recon_means = reconstructed.mean(axis=(1, 2))  # (5,)
    
    # Calculate correlation between original and reconstructed means
    correlation = np.corrcoef(orig_means, recon_means)[0, 1]
    
    # Check if any band has inverted brightness (negative correlation)
    band_inversions = []
    for i in range(5):
        orig_band = original[i]
        recon_band = reconstructed[i]
        band_corr = np.corrcoef(orig_band.flatten(), recon_band.flatten())[0, 1]
        if band_corr < -0.5:  # Strong negative correlation suggests inversion
            band_inversions.append(i)
    
    # Save diagnostic information
    diagnostic_info = {
        'sample_idx': idx,
        'original_range': [float(orig_min), float(orig_max)],
        'reconstructed_range': [float(recon_min), float(recon_max)],
        'original_means_per_band': orig_means.tolist(),
        'reconstructed_means_per_band': recon_means.tolist(),
        'overall_correlation': float(correlation),
        'inverted_bands': band_inversions,
        'potential_inversion': len(band_inversions) > 0
    }
    
    with open(os.path.join(outdir, f'sample_{idx}_normalization_check.json'), 'w') as f:
        json.dump(diagnostic_info, f, indent=2)
    
    # Print warnings if issues detected
    if len(band_inversions) > 0:
        print(f"WARNING: Sample {idx} has potential light/dark inversions in bands: {band_inversions}")
    
    if correlation < 0.5:
        print(f"WARNING: Sample {idx} has low correlation ({correlation:.3f}) between original and reconstructed brightness")
    
    return diagnostic_info

def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained multispectral VAE")
    parser.add_argument('--model_dir', type=str, required=True, help='Path to model directory (with config.json)')
    parser.add_argument('--val_file_list', type=str, required=True, help='Path to val/test file list')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='vae_eval_results')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Refactored: Use explicit adapter and backbone channel config fields for clarity
    model = AutoencoderKL.from_pretrained(
        args.model_dir,
        adapter_in_channels=5,  # Refactored: adapter input channels
        adapter_out_channels=5, # Refactored: adapter output channels
        backbone_in_channels=3, # Refactored: backbone input channels
        backbone_out_channels=3, # Refactored: backbone output channels
        latent_channels=16,
        adapter_placement="both", # match training
        use_spectral_attention=True,
        use_sam_loss=True,
        use_saturation_penalty=True,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
    )
    model = model.to(device)
    model.eval()

    # Load dataset with masks to enable separate foreground and background metric computation
    dataset = VAEMultispectralDataset(
        file_list_path=args.val_file_list,
        resolution=512,
        use_cache=False,
        return_mask=True  # Enable mask loading for leaf/background evaluation
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_mse_fg = []
    all_mse_bg = []
    all_mse_ov = []
    all_sam_fg = []
    all_sam_bg = []
    all_sam_ov = []
    all_ssim_fg = []
    all_ssim_bg = []
    all_ssim_ov = []
    all_signature_errors = []
    os.makedirs(args.output_dir, exist_ok=True)

    # After all batches are processed, compute mean reconstructed spectrum and deviation from reference
    recon_mean_spectra = []
    reference_signature = np.array([0.055, 0.12, 0.05, 0.31, 0.325], dtype=np.float32)
    for idx, (batch, mask) in enumerate(tqdm(loader, desc='Evaluating')):
        # Handle both tuple and single tensor returns from dataloader
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        batch = batch.to(device)
        mask = mask.to(device)
        if mask is None:
            print(f"[MASK WARNING] No mask provided for evaluation batch {idx}!")
        
        with torch.no_grad():
            recon = model(batch)
            # Extract tensor from DecoderOutput
            if hasattr(recon, "sample"):
                decoded_tensor = recon.sample
            else:
                decoded_tensor = recon

        # --- TEMP DEBUG: Check if model output is inverted relative to input ---
        # Find one valid pixel within the mask (foreground/leaf)
        mask_np_batch = mask.cpu().numpy()[0, 0]  # shape: (H, W)
        yx = np.argwhere(mask_np_batch > 0)
        if len(yx) > 0:
            y, x = yx[len(yx) // 2]  # select middle leaf pixel
        else:
            y, x = 25, 25  # fallback

        orig_pixel = batch[0, :, y, x].detach().cpu().numpy()
        recon_pixel = decoded_tensor[0, :, y, x].detach().cpu().numpy()

        print(f"[DEBUG] Spectral curve at pixel ({y},{x}):")
        print("Original (normalized):", np.round(orig_pixel, 3))
        print("Reconstructed (normalized):", np.round(recon_pixel, 3))

        corr = np.corrcoef(orig_pixel, recon_pixel)[0, 1]
        print(f"[DEBUG] Correlation between original and reconstruction: {corr:.3f}")

        if corr < -0.5:
            print("[WARNING] Strong negative correlation detected — potential spectral inversion.")

        # NOTE: Comment or remove this block once inversion issue is resolved.

        orig_np = batch.cpu().numpy()
        recon_np = decoded_tensor.cpu().numpy()
        mask_np = mask.cpu().numpy()
        
        # Visualize
        if idx < args.num_samples:
            plot_bands(orig_np[0], recon_np[0], args.output_dir, idx)
            plot_pseudo_rgb(orig_np[0], recon_np[0], args.output_dir, idx)
            points = plot_spectral_signature(orig_np[0], recon_np[0], mask_np, args.output_dir, idx)
            plot_error_maps(orig_np[0], recon_np[0], mask_np[0], args.output_dir, idx, points)
            # Check for normalization consistency and potential inversions
            check_normalization_consistency(orig_np[0], recon_np[0], args.output_dir, idx)
            # Compute and collect spectral signature errors for these points
            sig_errors = []
            for y, x in points:
                err = np.abs(orig_np[0, :, y, x] - recon_np[0, :, y, x])
                sig_errors.append(err)
            all_signature_errors.append(np.stack(sig_errors))  # shape: (n_points, 5)
        
        # Compute metrics for foreground (leaf), background, and overall
        mse_results = compute_bandwise_mse(orig_np, recon_np, mask_np)
        sam_results = compute_sam(orig_np, recon_np, mask_np)
        ssim_results = compute_ssim(orig_np, recon_np, mask_np)
        
        if mse_results['foreground'] is not None:
            all_mse_fg.append(mse_results['foreground'])
            all_mse_bg.append(mse_results['background'])
        all_mse_ov.append(mse_results['overall'])
        
        if sam_results['foreground'] is not None:
            all_sam_fg.append(sam_results['foreground']['mean_angle'])
            all_sam_bg.append(sam_results['background']['mean_angle'])
        all_sam_ov.append(sam_results['overall']['mean_angle'])
        
        if ssim_results['foreground'] is not None:
            all_ssim_fg.append(ssim_results['foreground'])
            all_ssim_bg.append(ssim_results['background'])
        all_ssim_ov.append(ssim_results['overall'])

        # Compute mean spectrum for this batch (foreground/leaf only)
        mask_sum = mask.sum(dim=(0,2,3)).cpu().numpy() + 1e-8
        recon_sum = (decoded_tensor * mask).sum(dim=(0,2,3)).cpu().numpy()
        recon_mean_spectrum = recon_sum / mask_sum
        recon_mean_spectra.append(recon_mean_spectrum)

    # Aggregate metrics by averaging over all batches
    all_mse_fg = np.stack(all_mse_fg) if all_mse_fg else None
    all_mse_bg = np.stack(all_mse_bg) if all_mse_bg else None
    all_mse_ov = np.stack(all_mse_ov)
    mean_mse_fg = np.nanmean(all_mse_fg, axis=0) if all_mse_fg is not None else None
    mean_mse_bg = np.nanmean(all_mse_bg, axis=0) if all_mse_bg is not None else None
    mean_mse_ov = np.nanmean(all_mse_ov, axis=0)

    all_sam_fg = np.array(all_sam_fg) if all_sam_fg else None
    all_sam_bg = np.array(all_sam_bg) if all_sam_bg else None
    all_sam_ov = np.array(all_sam_ov)
    mean_sam_fg = np.nanmean(all_sam_fg) if all_sam_fg is not None else None
    mean_sam_bg = np.nanmean(all_sam_bg) if all_sam_bg is not None else None
    mean_sam_ov = np.nanmean(all_sam_ov)

    all_ssim_fg = np.stack(all_ssim_fg) if all_ssim_fg else None
    all_ssim_bg = np.stack(all_ssim_bg) if all_ssim_bg else None
    all_ssim_ov = np.stack(all_ssim_ov)
    mean_ssim_fg = np.nanmean(all_ssim_fg, axis=0) if all_ssim_fg is not None else None
    mean_ssim_bg = np.nanmean(all_ssim_bg, axis=0) if all_ssim_bg is not None else None
    mean_ssim_ov = np.nanmean(all_ssim_ov, axis=0)

    # Save metrics with separate keys for foreground, background, and overall
    results = {
        'mean_mse_per_band_foreground': mean_mse_fg.tolist() if mean_mse_fg is not None else None,
        'mean_mse_per_band_background': mean_mse_bg.tolist() if mean_mse_bg is not None else None,
        'mean_mse_per_band_overall': mean_mse_ov.tolist(),
        'mean_sam_foreground': float(mean_sam_fg) if mean_sam_fg is not None else None,
        'mean_sam_background': float(mean_sam_bg) if mean_sam_bg is not None else None,
        'mean_sam_overall': float(mean_sam_ov),
        'mean_ssim_per_band_foreground': mean_ssim_fg.tolist() if mean_ssim_fg is not None else None,
        'mean_ssim_per_band_background': mean_ssim_bg.tolist() if mean_ssim_bg is not None else None,
        'mean_ssim_per_band_overall': mean_ssim_ov.tolist(),
        'mean_sam_per_band_foreground': sam_results['per_band_angle_foreground'],
        'mean_sam_per_band_background': sam_results['per_band_angle_background'],
        'mean_sam_per_band_overall': sam_results['per_band_angle_overall'],
    }
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print('Evaluation results:', results)
    if mean_ssim_fg is not None:
        print("Mean SSIM per band (foreground):", mean_ssim_fg)
        print("Mean SSIM per band (background):", mean_ssim_bg)
        print("Mean SSIM per band (overall):", mean_ssim_ov)
    else:
        print("SSIM not computed (skimage or pytorch_msssim not installed).")

    # Aggregate and plot spectral signature errors
    aggregate_spectral_errors(all_signature_errors, args.output_dir)

    # After all batches
    if recon_mean_spectra:
        eval_recon_mean_spectrum = np.mean(np.stack(recon_mean_spectra), axis=0)
        deviation = eval_recon_mean_spectrum - reference_signature
        print(f"\nMean reconstructed spectrum (eval set): {[f'{v:.4f}' for v in eval_recon_mean_spectrum]}")
        print(f"Reference signature: {[f'{v:.4f}' for v in reference_signature]}")
        print(f"Deviation from reference: {[f'{v:+.4f}' for v in deviation]}")
        # Save to file
        with open(os.path.join(args.output_dir, 'mean_spectrum_eval.json'), 'w') as f:
            json.dump({
                'mean_reconstructed_spectrum': eval_recon_mean_spectrum.tolist(),
                'reference_signature': reference_signature.tolist(),
                'deviation': deviation.tolist()
            }, f, indent=2)

if __name__ == '__main__':
    main()