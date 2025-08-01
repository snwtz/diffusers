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

import os
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import logging
from torchmetrics.image import SpectralAngleMapper


from diffusers.models.autoencoders.autoencoder_kl_multispectral_adapter import AutoencoderKLMultispectralAdapter
from vae_multispectral_dataloader import create_vae_dataloaders
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_dir: str, device: torch.device):
    """
    Load the trained multispectral VAE model.
    
    Args:
        model_dir: Path to the saved model directory
        device: Device to load the model on
        
    Returns:
        Loaded model in evaluation mode
    """
    logger.info(f"Loading model from: {model_dir}")
    
    try:
        model = AutoencoderKLMultispectralAdapter.from_pretrained(
            model_dir,
            adapter_in_channels=5,
            adapter_out_channels=5,
            backbone_in_channels=3,
            backbone_out_channels=3,
            adapter_placement="both",
            use_spectral_attention=True,
            use_sam_loss=True,
            use_saturation_penalty=True,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False,
        )
        model = model.to(device)
        model.eval()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def evaluate_sam_metrics(model, dataloader, device, num_samples=None):
    """
    Evaluate SAM metrics using torchmetrics implementation.
    
    Args:
        model: Trained VAE model
        dataloader: DataLoader for evaluation data
        device: Device for computation
        num_samples: Maximum number of samples to evaluate (None = all)
        
    Returns:
        Dictionary containing SAM evaluation results
    """
    if not TORCHMETRICS_AVAILABLE:
        raise ImportError("torchmetrics is required but not available")
    
    # Initialize torchmetrics SAM
    sam_metric = SpectralAngleMapper().to(device)
    
    # Storage for per-sample results
    sam_scores = []
    per_band_sam_scores = []  # For comparison with your per-band implementation
    
    logger.info("Starting SAM evaluation with torchmetrics...")
    
    with torch.no_grad():
        for batch_idx, (batch, mask) in enumerate(tqdm(dataloader, desc="Evaluating SAM")):
            if num_samples and batch_idx * dataloader.batch_size >= num_samples:
                break
                
            batch = batch.to(device)
            mask = mask.to(device)
            
            # Sanitize input (same as in training)
            if torch.isnan(batch).any():
                for band_idx in range(batch.shape[1]):
                    band = batch[:, band_idx]
                    nan_mask = torch.isnan(band)
                    if nan_mask.any():
                        band_mean = band[~nan_mask].mean() if (~nan_mask).any() else 0.0
                        batch[:, band_idx][nan_mask] = band_mean
            batch = torch.nan_to_num(batch, nan=0.0, posinf=1.0, neginf=-1.0)
            batch = torch.clamp(batch, min=-1.0, max=1.0)
            
            # Forward pass
            try:
                # Use the model's decode method directly for evaluation
                encoded = model.encode(batch)
                if hasattr(encoded, 'latent_dist'):
                    latent = encoded.latent_dist.mode()  # Use mode for deterministic evaluation
                else:
                    latent = encoded
                
                reconstruction = model.decode(latent, return_dict=False)
                if isinstance(reconstruction, tuple):
                    reconstruction = reconstruction[0]
                elif hasattr(reconstruction, 'sample'):
                    reconstruction = reconstruction.sample
                    
            except Exception as e:
                logger.warning(f"Forward pass failed for batch {batch_idx}: {e}")
                continue
            
            # Compute overall SAM using torchmetrics
            try:
                # Torchmetrics expects data in range [0, 1], so we need to convert from [-1, 1]
                batch_01 = (batch + 1.0) / 2.0
                reconstruction_01 = (reconstruction + 1.0) / 2.0
                
                # Apply mask to focus on leaf regions only
                if mask is not None:
                    mask_expanded = mask.expand_as(batch_01)
                    
                    # Only compute SAM on pixels with significant leaf coverage
                    leaf_pixels = mask_expanded.sum(dim=1) > 0.1  # At least 10% leaf coverage
                    
                    if leaf_pixels.any():
                        masked_batch = batch_01[leaf_pixels.unsqueeze(1).expand_as(batch_01)]
                        masked_recon = reconstruction_01[leaf_pixels.unsqueeze(1).expand_as(reconstruction_01)]
                        
                        # Reshape for torchmetrics: (N, C, H, W) where N is number of valid pixels
                        # torchmetrics expects each "image" to be a single pixel's spectrum
                        n_valid = leaf_pixels.sum().item()
                        if n_valid > 0:
                            # Reshape to treat each valid pixel as a separate "image"
                            pixel_batch = masked_batch.view(n_valid, batch.shape[1], 1, 1)
                            pixel_recon = masked_recon.view(n_valid, batch.shape[1], 1, 1)
                            
                            sam_score = sam_metric(pixel_recon, pixel_batch)
                            sam_scores.append(sam_score.item())
                else:
                    # No mask provided, compute on entire image
                    sam_score = sam_metric(reconstruction_01, batch_01)
                    sam_scores.append(sam_score.item())
                
                # Compute per-band analysis for comparison
                per_band_scores = []
                for band_idx in range(batch.shape[1]):
                    try:
                        # Extract single band and treat as grayscale "image"
                        band_orig = batch_01[:, band_idx:band_idx+1]  # Keep channel dim
                        band_recon = reconstruction_01[:, band_idx:band_idx+1]
                        
                        if mask is not None:
                            band_mask = mask
                            # Apply mask
                            masked_orig = band_orig * band_mask
                            masked_recon = band_recon * band_mask
                            
                            # Only compute if there are valid pixels
                            if (band_mask > 0.1).any():
                                band_sam = sam_metric(masked_recon, masked_orig)
                                per_band_scores.append(band_sam.item())
                            else:
                                per_band_scores.append(float('nan'))
                        else:
                            band_sam = sam_metric(band_recon, band_orig)
                            per_band_scores.append(band_sam.item())
                    except Exception as e:
                        logger.warning(f"Per-band SAM failed for band {band_idx}: {e}")
                        per_band_scores.append(float('nan'))
                
                per_band_sam_scores.append(per_band_scores)
                
            except Exception as e:
                logger.warning(f"SAM computation failed for batch {batch_idx}: {e}")
                continue
    
    # Compute statistics
    if sam_scores:
        results = {
            'overall_sam': {
                'mean': np.mean(sam_scores),
                'std': np.std(sam_scores),
                'min': np.min(sam_scores),
                'max': np.max(sam_scores),
                'median': np.median(sam_scores),
                'n_samples': len(sam_scores)
            }
        }
        
        # Per-band statistics
        if per_band_sam_scores:
            per_band_array = np.array(per_band_sam_scores)
            per_band_means = np.nanmean(per_band_array, axis=0)
            per_band_stds = np.nanstd(per_band_array, axis=0)
            
            results['per_band_sam'] = {
                'means': per_band_means.tolist(),
                'stds': per_band_stds.tolist(),
                'band_names': ['Blue (474.73nm)', 'Green (538.71nm)', 'Red (650.665nm)', 
                              'Red-edge (730.635nm)', 'NIR (850.59nm)']
            }
        
        logger.info(f"SAM Evaluation Results:")
        logger.info(f"  Overall SAM: {results['overall_sam']['mean']:.4f} ± {results['overall_sam']['std']:.4f} rad")
        logger.info(f"  Overall SAM (degrees): {results['overall_sam']['mean'] * 180/np.pi:.2f} ± {results['overall_sam']['std'] * 180/np.pi:.2f}°")
        
        if 'per_band_sam' in results:
            logger.info("  Per-band SAM (radians):")
            for i, (name, mean_val, std_val) in enumerate(zip(
                results['per_band_sam']['band_names'],
                results['per_band_sam']['means'],
                results['per_band_sam']['stds']
            )):
                logger.info(f"    {name}: {mean_val:.4f} ± {std_val:.4f}")
        
        return results
    else:
        logger.error("No valid SAM scores computed")
        return {}

def compare_with_custom_sam(original, reconstructed, mask=None):
    """
    Compare torchmetrics SAM with a custom implementation for validation.
    
    Args:
        original: Original tensor in [-1, 1] range
        reconstructed: Reconstructed tensor in [-1, 1] range
        mask: Optional mask tensor
        
    Returns:
        Dictionary with comparison results
    """
    if not TORCHMETRICS_AVAILABLE:
        return {}
    
    device = original.device
    sam_metric = SpectralAngleMapper().to(device)
    
    # Convert to [0, 1] for torchmetrics
    orig_01 = (original + 1.0) / 2.0
    recon_01 = (reconstructed + 1.0) / 2.0
    
    # Torchmetrics SAM
    torchmetrics_sam = sam_metric(recon_01, orig_01).item()
    
    # Custom SAM implementation (matching your training code)
    def custom_sam_loss(orig, recon):
        # Normalize to unit vectors
        orig_norm = torch.nn.functional.normalize(orig, p=2, dim=1)
        recon_norm = torch.nn.functional.normalize(recon, p=2, dim=1)
        
        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(orig_norm, recon_norm, dim=1)
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Convert to angle
        angle = torch.acos(cos_sim)
        return angle.mean().item()
    
    custom_sam = custom_sam_loss(original, reconstructed)
    
    return {
        'torchmetrics_sam': torchmetrics_sam,
        'custom_sam': custom_sam,
        'difference': abs(torchmetrics_sam - custom_sam),
        'relative_error': abs(torchmetrics_sam - custom_sam) / max(torchmetrics_sam, 1e-8)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate VAE with torchmetrics SpectralAngleMapper")
    parser.add_argument('--model_dir', type=str, required=True, 
                       help='Path to trained model directory')
    parser.add_argument('--val_file_list', type=str, required=True,
                       help='Path to validation file list')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (None = all)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save results')
    parser.add_argument('--compare_implementations', action='store_true',
                       help='Compare torchmetrics with custom SAM implementation')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.model_dir).parent / "torchmetrics_sam_evaluation")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check torchmetrics availability
    if not TORCHMETRICS_AVAILABLE:
        logger.error("torchmetrics is not available. Install with: pip install torchmetrics")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_dir, device)
    
    # Create dataloader
    try:
        _, val_loader = create_vae_dataloaders(
            train_list_path=args.val_file_list,  # Use val list for both (we only need val)
            val_list_path=args.val_file_list,
            batch_size=args.batch_size,
            resolution=512,
            num_workers=0,  # Avoid multiprocessing issues
            use_cache=False,
            return_mask=True
        )
        logger.info(f"Created dataloader with {len(val_loader.dataset)} validation samples")
    except Exception as e:
        logger.error(f"Failed to create dataloader: {e}")
        return
    
    # Evaluate SAM metrics
    try:
        results = evaluate_sam_metrics(model, val_loader, device, args.num_samples)
        
        # Save results
        results_file = os.path.join(args.output_dir, 'torchmetrics_sam_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {results_file}")
        
        # Compare implementations if requested
        if args.compare_implementations and results:
            logger.info("\nComparing torchmetrics with custom SAM implementation...")
            
            # Test on a single batch
            for batch, mask in val_loader:
                batch = batch.to(device)
                mask = mask.to(device)
                
                # Sanitize input
                batch = torch.nan_to_num(batch, nan=0.0, posinf=1.0, neginf=-1.0)
                batch = torch.clamp(batch, min=-1.0, max=1.0)
                
                # Get reconstruction
                with torch.no_grad():
                    encoded = model.encode(batch)
                    if hasattr(encoded, 'latent_dist'):
                        latent = encoded.latent_dist.mode()
                    else:
                        latent = encoded
                    reconstruction = model.decode(latent, return_dict=False)
                    if isinstance(reconstruction, tuple):
                        reconstruction = reconstruction[0]
                    elif hasattr(reconstruction, 'sample'):
                        reconstruction = reconstruction.sample
                
                # Compare implementations
                comparison = compare_with_custom_sam(batch, reconstruction, mask)
                
                logger.info("Implementation Comparison:")
                logger.info(f"  Torchmetrics SAM: {comparison['torchmetrics_sam']:.6f} rad")
                logger.info(f"  Custom SAM: {comparison['custom_sam']:.6f} rad")
                logger.info(f"  Absolute difference: {comparison['difference']:.6f} rad")
                logger.info(f"  Relative error: {comparison['relative_error']:.2%}")
                
                # Save comparison
                comparison_file = os.path.join(args.output_dir, 'sam_implementation_comparison.json')
                with open(comparison_file, 'w') as f:
                    json.dump(comparison, f, indent=2)
                
                break  # Only test on first batch
                
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    
    logger.info("Evaluation completed successfully!")

if __name__ == '__main__':
    main() 