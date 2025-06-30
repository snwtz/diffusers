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
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None

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
    points = [(np.random.randint(0, h), np.random.randint(0, w)) for _ in range(n_points)]
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

def compute_sam(original, reconstructed, eps=1e-8):
    # original/reconstructed: (B, 5, H, W)
    orig_flat = original.reshape(original.shape[0], 5, -1)
    recon_flat = reconstructed.reshape(reconstructed.shape[0], 5, -1)
    dot = (orig_flat * recon_flat).sum(1)
    norm1 = np.linalg.norm(orig_flat, axis=1)
    norm2 = np.linalg.norm(recon_flat, axis=1)
    cos = dot / (norm1 * norm2 + eps)
    cos = np.clip(cos, -1, 1)
    angle = np.arccos(cos)
    return np.nanmean(angle)

def compute_bandwise_mse(original, reconstructed):
    # original/reconstructed: (B, 5, H, W)
    mse = ((original - reconstructed) ** 2).mean(axis=(0,2,3))
    return mse

def compute_ssim(original, reconstructed):
    # original/reconstructed: (B, 5, H, W)
    if ssim is None:
        return None
    ssim_scores = []
    for band in range(5):
        band_scores = []
        for i in range(original.shape[0]):
            score = ssim(original[i, band], reconstructed[i, band], data_range=2.0)
            band_scores.append(score)
        ssim_scores.append(np.mean(band_scores))
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

    dataset = VAEMultispectralDataset(
        file_list_path=args.val_file_list,
        resolution=512,
        use_cache=False,
        return_mask=False
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_mse = []
    all_sam = []
    all_ssim = []
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, batch in enumerate(tqdm(loader, desc='Evaluating')):
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        batch = batch.to(device)
        with torch.no_grad():
            recon, _ = model(batch)
        orig_np = batch.cpu().numpy()
        recon_np = recon.cpu().numpy()
        # Visualize
        if idx < args.num_samples:
            plot_bands(orig_np[0], recon_np[0], args.output_dir, idx)
            plot_spectral_signature(orig_np[0], recon_np[0], args.output_dir, idx)
        # Metrics
        all_mse.append(compute_bandwise_mse(orig_np, recon_np))
        all_sam.append(compute_sam(orig_np, recon_np))
        if ssim is not None:
            all_ssim.append(compute_ssim(orig_np, recon_np))
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

if __name__ == '__main__':
    main() 