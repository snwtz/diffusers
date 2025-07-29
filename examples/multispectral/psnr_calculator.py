#!/usr/bin/env python3
"""
Standalone PSNR Calculator for Multispectral VAE Reconstructions

This script calculates Peak Signal-to-Noise Ratio (PSNR) for multispectral VAE 
reconstructions to assess reconstruction quality. Designed specifically for the 
multispectral VAE research setup that processes 5-channel hyperspectral data.

Key Features:
- Compatible with hyperspectral compression (bands 9, 18, 32, 42, 55)
- Supports TIFF file loading via rasterio
- Handles [-1, 1] normalized data range
- Leaf-focused evaluation using background masks
- Per-band and overall PSNR calculation
- Compatible with VAE dataloader preprocessing

Usage:
    # Single file comparison
python psnr_calculator.py --original original.tiff --reconstructed reconstructed.pt

# Batch processing
python psnr_calculator.py --original_dir /path/to/originals --reconstructed_dir /path/to/reconstructions

"""

import torch
import numpy as np
import rasterio
import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
from torchmetrics.image import PeakSignalNoiseRatio
from pathlib import Path
import argparse
from typing import Union, List, Dict, Tuple, Optional
import logging
import json
from tqdm import tqdm
import torch.nn.functional as F

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultispectralPSNRCalculator:
    """
    PSNR calculator specifically designed for multispectral VAE reconstructions.
    
    Handles the unique characteristics of multispectral plant data:
    - 5-channel hyperspectral compression (bands 9, 18, 32, 42, 55)
    - TIFF file loading with rasterio
    - [-1, 1] normalized data range
    - Background masking for leaf-focused evaluation
    - Per-band and overall PSNR metrics
    """
    
    # Define the specific bands used in the VAE (1-based indexing for rasterio.read)
    REQUIRED_BANDS = [9, 18, 32, 42, 55]  # Same as VAE dataloader
    
    # Wavelength mapping for interpretability
    WAVELENGTHS = {
        0: 474.73,  # Band 9: Blue - chlorophyll absorption
        1: 538.71,  # Band 18: Green - healthy vegetation  
        2: 650.665, # Band 32: Red - chlorophyll content
        3: 730.635, # Band 42: Red-edge - stress detection
        4: 850.59   # Band 55: NIR - leaf health
    }
    
    def __init__(self, resolution: int = 512, device: str = "cpu"):
        """
        Initialize the PSNR calculator.
        
        Args:
            resolution: Expected image resolution (default: 512)
            device: Device for tensor operations (default: "cpu")
        """
        self.resolution = resolution
        self.device = device
        
        # Initialize PSNR metric for [-1, 1] range
        self.psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(device)
        
        logger.info(f"Initialized PSNR calculator for {resolution}x{resolution} images")
        logger.info(f"Using bands: {self.REQUIRED_BANDS} (wavelengths: {list(self.WAVELENGTHS.values())})")
    
    def load_tiff_image(self, tiff_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess a multispectral TIFF image (exact same as VAE dataloader).
        
        Args:
            tiff_path: Path to the TIFF file
            
        Returns:
            Tuple of (image_tensor, mask_tensor)
            - image_tensor: (5, H, W) normalized to [-1, 1]
            - mask_tensor: (1, H, W) binary mask (1=leaf, 0=background)
        """
        try:
            with rasterio.open(tiff_path) as src:
                # Read required bands
                image = src.read(self.REQUIRED_BANDS)  # Shape: (5, height, width)
                
                # Generate background mask from NaN values (same as VAE dataloader)
                background_mask = np.isnan(image[0]).astype(np.float32)
                leaf_mask = 1 - background_mask  # 1 for leaf, 0 for background
                
                # Convert to float32 and normalize (same as VAE dataloader)
                image = image.astype(np.float32)
                normalized_image = np.zeros_like(image)
                
                # Fill NaN with per-band mean and normalize
                for i in range(5):
                    band = image[i]
                    nan_mask = np.isnan(band)
                    mean_val = np.nanmean(band)
                    band[nan_mask] = mean_val
                    normalized_image[i] = self._normalize_channel(band)
                
                # Convert to tensor
                image_tensor = torch.from_numpy(normalized_image)
                if torch.isnan(image_tensor).any():
                    logger.info(f"[Sanitize] Replacing NaNs in input tensor with 0.0 to avoid propagation into model.")
                    image_tensor = torch.nan_to_num(image_tensor, nan=0.0)
                
                # Compute fill_value as mean of valid (foreground) pixels per band for padding
                # This prevents artificial edges by matching the padded value to the leaf distribution
                # (fixes "bounding box" style artifacts) - EXACT SAME AS VAE DATALOADER
                foreground_mask = torch.from_numpy(leaf_mask).unsqueeze(0).bool()  # shape: (1, H, W)
                # Compute per-band mean for valid (foreground) pixels
                per_band_means = []
                for b in range(image_tensor.shape[0]):
                    band_pixels = image_tensor[b][foreground_mask[0]]
                    band_mean = band_pixels.mean().item() if band_pixels.numel() > 0 else 0.0
                    per_band_means.append(band_mean)
                # Always use the average of per-band means as a scalar fill value for F.pad
                fill_value = float(np.mean(per_band_means))
                
                # Pad to square before resizing
                image_tensor = self._pad_to_square(image_tensor, fill_value=fill_value)
                
                # Now resize to (resolution, resolution) if needed (should be square already)
                if image_tensor.shape[1] != self.resolution or image_tensor.shape[2] != self.resolution:
                    image_tensor = F.interpolate(
                        image_tensor.unsqueeze(0),
                        size=(self.resolution, self.resolution),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                
                # Pad and resize mask
                mask_tensor = torch.from_numpy(leaf_mask).unsqueeze(0)  # (1, H, W)
                mask_tensor = self._pad_to_square(mask_tensor, fill_value=0.0)
                if mask_tensor.shape[1] != self.resolution or mask_tensor.shape[2] != self.resolution:
                    mask_tensor = F.interpolate(
                        mask_tensor.unsqueeze(0),
                        size=(self.resolution, self.resolution),
                        mode='nearest'
                    ).squeeze(0)
                
                # All non-leaf (background) regions are zeroed out after resizing.
                # This avoids any contribution from padding or masked areas during training, as background has zero importance.
                # Explicitly zero out background regions (non-leaf) after resizing - EXACT SAME AS VAE DATALOADER
                image_tensor = image_tensor * mask_tensor
                
                return image_tensor, mask_tensor
                
        except Exception as e:
            logger.error(f"Failed to load TIFF file {tiff_path}: {e}")
            raise
    
    def _normalize_channel(self, channel_data: np.ndarray) -> np.ndarray:
        """
        Per-channel normalization to [-1, 1] range (same as VAE dataloader).
        
        Args:
            channel_data: Input channel data
            
        Returns:
            Normalized channel data in [-1, 1] range
        """
        valid_mask = ~np.isnan(channel_data)
        
        if not np.any(valid_mask):
            return np.zeros_like(channel_data, dtype=np.float32)
        
        min_val = np.nanmin(channel_data)
        max_val = np.nanmax(channel_data)
        
        if max_val == min_val:
            return np.zeros_like(channel_data, dtype=np.float32)
        
        # Normalize to [0, 1] then scale to [-1, 1]
        normalized = np.full_like(channel_data, np.nan, dtype=np.float32)
        normalized[valid_mask] = (channel_data[valid_mask] - min_val) / (max_val - min_val)
        normalized[valid_mask] = 2 * normalized[valid_mask] - 1
        
        return normalized
    
    def _pad_and_resize(self, img: torch.Tensor, is_mask: bool = False) -> torch.Tensor:
        """
        Pad to square and resize to target resolution (same as VAE dataloader).
        
        Args:
            img: Input tensor (C, H, W)
            is_mask: Whether this is a mask tensor
            
        Returns:
            Padded and resized tensor
        """
        c, h, w = img.shape
        size = max(h, w, self.resolution)
        
        # Pad to square
        pad_h = size - h
        pad_w = size - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        
        if is_mask:
            fill_value = 0.0
        else:
            # Use mean of valid pixels as fill value (same as VAE dataloader)
            foreground_mask = (img[0] != 0).bool() if img.shape[0] > 0 else torch.ones_like(img[0]).bool()
            if foreground_mask.any():
                fill_value = float(img[0][foreground_mask].mean().item())
            else:
                fill_value = 0.0
        
        img = F.pad(img, padding, value=fill_value)
        
        # Resize if needed
        if img.shape[1] != self.resolution or img.shape[2] != self.resolution:
            mode = 'nearest' if is_mask else 'bilinear'
            img = F.interpolate(
                img.unsqueeze(0),
                size=(self.resolution, self.resolution),
                mode=mode,
                align_corners=False
            ).squeeze(0)
        
        return img
    
    def load_reconstructed_tensor(self, tensor_path: str) -> torch.Tensor:
        """
        Load a reconstructed tensor from file.
        
        Args:
            tensor_path: Path to the tensor file (.pt, .pth, or .npy)
            
        Returns:
            Reconstructed tensor (5, H, W)
        """
        tensor_path = Path(tensor_path)
        
        if tensor_path.suffix in ['.pt', '.pth']:
            tensor = torch.load(tensor_path, map_location=self.device)
        elif tensor_path.suffix == '.npy':
            tensor = torch.from_numpy(np.load(tensor_path)).to(self.device)
        else:
            raise ValueError(f"Unsupported tensor format: {tensor_path.suffix}")
        
        # Handle different tensor formats
        if isinstance(tensor, dict):
            if 'sample' in tensor:
                tensor = tensor['sample']
            elif 'reconstruction' in tensor:
                tensor = tensor['reconstruction']
            else:
                raise ValueError(f"Unknown tensor dict format: {list(tensor.keys())}")
        
        # Ensure correct shape and move to device
        if tensor.dim() == 4:  # (B, C, H, W)
            tensor = tensor.squeeze(0)  # Remove batch dimension
        elif tensor.dim() != 3:
            raise ValueError(f"Expected 3D tensor (C, H, W), got shape {tensor.shape}")
        
        if tensor.shape[0] != 5:
            raise ValueError(f"Expected 5 channels, got {tensor.shape[0]}")
        
        return tensor.float()
    
    def calculate_psnr(self, original: torch.Tensor, reconstructed: torch.Tensor, 
                      mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Calculate PSNR between original and reconstructed images.
        
        Args:
            original: Original image tensor (5, H, W)
            reconstructed: Reconstructed image tensor (5, H, W)
            mask: Optional binary mask (1, H, W) for leaf regions
            
        Returns:
            Dictionary containing PSNR metrics
        """
        # Move tensors to device
        original = original.to(self.device)
        reconstructed = reconstructed.to(self.device)
        
        if mask is not None:
            mask = mask.to(self.device)
        
        results = {}
        
        # Overall PSNR (all channels)
        if mask is not None:
            # Apply mask for leaf-focused evaluation
            masked_original = original * mask
            masked_reconstructed = reconstructed * mask
            overall_psnr = self.psnr_metric(masked_reconstructed, masked_original)
        else:
            overall_psnr = self.psnr_metric(reconstructed, original)
        
        results['overall_psnr'] = overall_psnr.item()
        
        # Per-band PSNR
        per_band_psnr = []
        for i in range(5):
            if mask is not None:
                band_original = original[i:i+1] * mask
                band_reconstructed = reconstructed[i:i+1] * mask
                band_psnr = self.psnr_metric(band_reconstructed, band_original)
            else:
                band_original = original[i:i+1]
                band_reconstructed = reconstructed[i:i+1]
                band_psnr = self.psnr_metric(band_reconstructed, band_original)
            
            wavelength = self.WAVELENGTHS[i]
            results[f'band_{i+1}_psnr'] = band_psnr.item()
            results[f'wavelength_{wavelength:.1f}nm_psnr'] = band_psnr.item()
            per_band_psnr.append(band_psnr.item())
        
        # Average per-band PSNR
        results['mean_per_band_psnr'] = np.mean(per_band_psnr)
        results['std_per_band_psnr'] = np.std(per_band_psnr)
        
        # Additional statistics
        if mask is not None:
            mask_coverage = mask.mean().item()
            results['mask_coverage'] = mask_coverage
            results['evaluation_mode'] = 'leaf_focused'
        else:
            results['evaluation_mode'] = 'full_image'
        
        return results
    
    def calculate_batch_psnr(self, original_dir: str, reconstructed_dir: str, 
                           output_file: str = "psnr_results.json") -> Dict[str, any]:
        """
        Calculate PSNR for a batch of images.
        
        Args:
            original_dir: Directory containing original TIFF files
            reconstructed_dir: Directory containing reconstructed tensors
            output_file: Output JSON file for results
            
        Returns:
            Dictionary containing batch PSNR results
        """
        original_dir = Path(original_dir)
        reconstructed_dir = Path(reconstructed_dir)
        
        if not original_dir.exists():
            raise FileNotFoundError(f"Original directory not found: {original_dir}")
        if not reconstructed_dir.exists():
            raise FileNotFoundError(f"Reconstructed directory not found: {reconstructed_dir}")
        
        # Find matching files
        tiff_files = list(original_dir.glob("*.tiff")) + list(original_dir.glob("*.tif"))
        tensor_files = list(reconstructed_dir.glob("*.pt")) + list(reconstructed_dir.glob("*.pth")) + list(reconstructed_dir.glob("*.npy"))
        
        if not tiff_files:
            raise FileNotFoundError(f"No TIFF files found in {original_dir}")
        if not tensor_files:
            raise FileNotFoundError(f"No tensor files found in {reconstructed_dir}")
        
        logger.info(f"Found {len(tiff_files)} TIFF files and {len(tensor_files)} tensor files")
        
        # Match files by name (without extension)
        file_pairs = []
        for tiff_file in tiff_files:
            base_name = tiff_file.stem
            matching_tensor = None
            
            for tensor_file in tensor_files:
                if tensor_file.stem == base_name:
                    matching_tensor = tensor_file
                    break
            
            if matching_tensor:
                file_pairs.append((tiff_file, matching_tensor))
            else:
                logger.warning(f"No matching tensor found for {tiff_file}")
        
        if not file_pairs:
            raise ValueError("No matching file pairs found")
        
        logger.info(f"Processing {len(file_pairs)} file pairs")
        
        # Calculate PSNR for each pair
        all_results = []
        overall_stats = {
            'overall_psnr': [],
            'mean_per_band_psnr': [],
            'mask_coverage': [],
            'band_1_psnr': [], 'band_2_psnr': [], 'band_3_psnr': [], 
            'band_4_psnr': [], 'band_5_psnr': []
        }
        
        for tiff_file, tensor_file in tqdm(file_pairs, desc="Calculating PSNR"):
            try:
                # Load original and reconstructed
                original, mask = self.load_tiff_image(str(tiff_file))
                reconstructed = self.load_reconstructed_tensor(str(tensor_file))
                
                # Calculate PSNR
                results = self.calculate_psnr(original, reconstructed, mask)
                results['original_file'] = str(tiff_file)
                results['reconstructed_file'] = str(tensor_file)
                
                all_results.append(results)
                
                # Accumulate statistics
                for key in overall_stats:
                    if key in results:
                        overall_stats[key].append(results[key])
                
            except Exception as e:
                logger.error(f"Failed to process {tiff_file}: {e}")
                continue
        
        # Calculate summary statistics
        summary = {
            'num_files_processed': len(all_results),
            'overall_mean_psnr': np.mean(overall_stats['overall_psnr']),
            'overall_std_psnr': np.std(overall_stats['overall_psnr']),
            'mean_per_band_psnr': np.mean(overall_stats['mean_per_band_psnr']),
            'std_per_band_psnr': np.std(overall_stats['mean_per_band_psnr']),
            'mean_mask_coverage': np.mean(overall_stats['mask_coverage']),
            'per_band_summary': {}
        }
        
        # Per-band summary
        for i in range(1, 6):
            band_key = f'band_{i}_psnr'
            if band_key in overall_stats:
                summary['per_band_summary'][f'band_{i}'] = {
                    'mean_psnr': np.mean(overall_stats[band_key]),
                    'std_psnr': np.std(overall_stats[band_key]),
                    'wavelength_nm': self.WAVELENGTHS[i-1]
                }
        
        # Save results
        output_data = {
            'summary': summary,
            'individual_results': all_results,
            'metadata': {
                'resolution': self.resolution,
                'required_bands': self.REQUIRED_BANDS,
                'wavelengths': self.WAVELENGTHS,
                'data_range': '[-1, 1]',
                'evaluation_mode': 'leaf_focused' if overall_stats['mask_coverage'] else 'full_image'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        logger.info(f"Summary: Mean PSNR = {summary['overall_mean_psnr']:.2f} ± {summary['overall_std_psnr']:.2f}")
        
        return output_data


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Calculate PSNR for multispectral VAE reconstructions")
    parser.add_argument("--original", type=str, help="Path to original TIFF file")
    parser.add_argument("--reconstructed", type=str, help="Path to reconstructed tensor file")
    parser.add_argument("--original_dir", type=str, help="Directory containing original TIFF files")
    parser.add_argument("--reconstructed_dir", type=str, help="Directory containing reconstructed tensor files")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution (default: 512)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for computation (default: cpu)")
    parser.add_argument("--output", type=str, default="psnr_results.json", help="Output JSON file (default: psnr_results.json)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.original and args.reconstructed:
        # Single file comparison
        if not Path(args.original).exists():
            raise FileNotFoundError(f"Original file not found: {args.original}")
        if not Path(args.reconstructed).exists():
            raise FileNotFoundError(f"Reconstructed file not found: {args.reconstructed}")
        
        calculator = MultispectralPSNRCalculator(resolution=args.resolution, device=args.device)
        
        # Load files
        original, mask = calculator.load_tiff_image(args.original)
        reconstructed = calculator.load_reconstructed_tensor(args.reconstructed)
        
        # Calculate PSNR
        results = calculator.calculate_psnr(original, reconstructed, mask)
        
        # Print results
        print(f"\nPSNR Results for {Path(args.original).name}:")
        print(f"Overall PSNR: {results['overall_psnr']:.2f} dB")
        print(f"Mean per-band PSNR: {results['mean_per_band_psnr']:.2f} ± {results['std_per_band_psnr']:.2f} dB")
        print(f"Evaluation mode: {results['evaluation_mode']}")
        
        if 'mask_coverage' in results:
            print(f"Mask coverage: {results['mask_coverage']:.3f}")
        
        print("\nPer-band PSNR:")
        for i in range(1, 6):
            wavelength = calculator.WAVELENGTHS[i-1]
            psnr = results[f'band_{i}_psnr']
            print(f"  Band {i} ({wavelength:.1f}nm): {psnr:.2f} dB")
        
        # Save results
        output_data = {
            'single_file_results': results,
            'metadata': {
                'original_file': args.original,
                'reconstructed_file': args.reconstructed,
                'resolution': args.resolution,
                'required_bands': calculator.REQUIRED_BANDS,
                'wavelengths': calculator.WAVELENGTHS,
                'data_range': '[-1, 1]'
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {args.output}")
        
    elif args.original_dir and args.reconstructed_dir:
        # Batch processing
        calculator = MultispectralPSNRCalculator(resolution=args.resolution, device=args.device)
        results = calculator.calculate_batch_psnr(args.original_dir, args.reconstructed_dir, args.output)
        
        # Print summary
        summary = results['summary']
        print(f"\nBatch PSNR Summary ({summary['num_files_processed']} files):")
        print(f"Overall mean PSNR: {summary['overall_mean_psnr']:.2f} ± {summary['overall_std_psnr']:.2f} dB")
        print(f"Mean per-band PSNR: {summary['mean_per_band_psnr']:.2f} ± {summary['std_per_band_psnr']:.2f} dB")
        print(f"Mean mask coverage: {summary['mean_mask_coverage']:.3f}")
        
        print("\nPer-band summary:")
        for band, stats in summary['per_band_summary'].items():
            wavelength = stats['wavelength_nm']
            print(f"  {band} ({wavelength:.1f}nm): {stats['mean_psnr']:.2f} ± {stats['std_psnr']:.2f} dB")
        
    else:
        parser.error("Must specify either --original/--reconstructed for single file or --original_dir/--reconstructed_dir for batch processing")


if __name__ == "__main__":
    main() 