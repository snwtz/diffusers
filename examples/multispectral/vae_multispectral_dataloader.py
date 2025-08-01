"""
Multispectral Image Dataloader for VAE Training
===============================================

This module implements a specialized dataloader for multispectral TIFF images,
optimized for training a VAE on multispectral plant data. It extends the base
multispectral dataloader to support train/val splits via file lists.

USAGE:
------
# Create dataloaders for training
train_loader, val_loader = create_vae_dataloaders(
    train_list_path="path/to/train_files.txt",
    val_list_path="path/to/val_files.txt",
    batch_size=8,
    resolution=512,
    num_workers=4,
    use_cache=True,
    return_mask=True
)

CONFIGURATION:
--------------
- Input: Hyperspectral TIFF files with at least 55 bands
- Output: 5-channel tensors (bands 9, 18, 32, 42, 55) normalized to [-1, 1]
- Background: NaN values represent background regions, excluded from training
- Mask: Binary mask (1=leaf, 0=background) for loss computation

Key Features:
- Support for train/val splits via file lists
- Optimized for VAE training
- Memory-efficient loading with optional caching
- Spectral fidelity preservation
- Robust validation
- Background masking using NaN values
- Leaf-focused feature learning

Implementation Notes:
--------------------
1. Background Handling:
   - NaN values in TIFF files represent background (cut-out regions)
   - These regions are masked out during training
   - Model focuses on leaf features

   The loss function:
    • Applies full loss (100%) on leaf regions (mask == 1).
    • Applies no loss (0%) on background (mask == 0), as cut-out images contain no true background. Padding is excluded from loss via masking.

2. Data Processing:
   The training pipeline handles 5 biologically relevant spectral bands:
   - Band 9 (474.73nm): Blue - captures chlorophyll absorption
   - Band 18 (538.71nm): Green - reflects well in healthy vegetation
   - Band 32 (650.665nm): Red - sensitive to chlorophyll content
   - Band 42 (730.635nm): Red-edge - sensitive to stress and early disease
   - Band 55 (850.59nm): NIR - strong reflectance in healthy leaves

3. Input Sanitization:
   - NaN values are preserved for background masking until final tensor conversion
   - Just before model input, NaNs in the input tensor are replaced with 0.0
   - This prevents NaN propagation into model layers and loss functions
   - Maintains compatibility with SD-style pipelines that expect dense tensors
"""

import os
import torch
import numpy as np
import rasterio
import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VAEMultispectralDataset(Dataset):
    """
    Dataset class for loading and preprocessing multispectral TIFF images for VAE training.
    Handles 5-channel data by selecting specific bands (9, 18, 32, 42, 55) from input TIFFs.
    Supports loading from file lists for train/val splits.

    Implementation Details:
    ---------------------
    1. Band Selection:
       - Uses specific bands optimized for plant analysis
       - Maintains spectral relationships between bands
       - Ensures consistent input dimensions for VAE

    2. Background Handling:
       - NaN values represent background (cut-out regions)
       - Generates binary mask (1 for leaf, 0 for background)
       - Background is excluded from loss computation (mask == 0); only leaf regions contribute to learning.

    3. Normalization:
       - Per-channel normalization to [-1, 1] range
       - Only normalizes valid (non-NaN) regions
       - Preserves spectral relationships
       - Required for VAE training stability
    5. Reproducibility:
       - File list is expected to be pre-generated and deterministic for scientific reproducibility
    """

    # Define the specific bands to use (1-based indexing for rasterio.read)
    # These bands are selected for optimal plant analysis:
    # - Band 9 (474.73nm): Blue - captures chlorophyll absorption
    # - Band 18 (538.71nm): Green - reflects well in healthy vegetation
    # - Band 32 (650.665nm): Red - sensitive to chlorophyll content
    # - Band 42 (730.635nm): Red-edge - sensitive to stress and early disease
    # - Band 55 (850.59nm): NIR - strong reflectance in healthy leaves
    REQUIRED_BANDS = [9, 18, 32, 42, 55]  # 1-based indices for rasterio.read

    def __init__(
        self,
        file_list_path: str,
        resolution: int = 512,
        transform: Optional[transforms.Compose] = None,
        use_cache: bool = True,
        return_mask: bool = True  # New parameter to control mask return
    ):
        """
        Initialize the dataset.

        Args:
            file_list_path: Path to train_files.txt or val_files.txt
            resolution: Target resolution for images (default: 512)
            transform: Additional transforms to apply (must be NaN-safe)
            use_cache: Whether to cache loaded images in memory
            return_mask: Whether to return the background mask

        Implementation Notes:
        -------------------
        1. File List Handling:
           - Supports train/val splits via file lists
           - Validates all files on initialization
           - Ensures consistent data access

        2. Background Masking:
           - NaN values represent background
           - Generates binary mask for leaf regions
           - Masks out background during training
           - Optional mask return for loss computation

        3. Caching Strategy:
           - Optional in-memory caching
           - Reduces disk I/O during training
           - Memory-efficient for large datasets
        """
        self.file_list_path = Path(file_list_path)
        if not self.file_list_path.exists():
            raise FileNotFoundError(f"File list not found: {file_list_path}")

        # Read file paths from the list
        with open(self.file_list_path, 'r') as f:
            self.image_paths = [Path(line.strip()) for line in f.readlines()]

        self.resolution = resolution
        self.transform = transform
        self.use_cache = use_cache
        self.return_mask = return_mask
        self.cache = {} if use_cache else None

        # Validate all images on initialization
        # This ensures that all files are valid before training starts, preventing runtime interruptions and ensuring reproducibility.
        self._validate_all_images()
        logger.info(f"Loaded {len(self.image_paths)} files from {file_list_path}")

    def _validate_all_images(self):
        """
        Validate that all images have at least 55 bands and correct data range.
        Handles NaN values as background.

        Implementation Notes:
        -------------------
        1. Band Count Validation:
           - Ensures all images have required bands (critical for spectral consistency)
           - Prevents runtime errors during training
           - Maintains data consistency

        2. Data Range Validation:
           - Ensures valid data is properly normalized
           - Ignores NaN values (background)
           - Prevents training instability
           - Maintains VAE compatibility
           - Data range warnings are not fatal; normalization will correct them, but they are logged for data quality monitoring.

        3. Error Handling:
           - Comprehensive error messages
           - Early failure for invalid data
           - Clear debugging information
           - Proper NaN handling
        """
        for path in self.image_paths:
            try:
                with rasterio.open(path) as src:
                    # Check band count
                    if src.count < 55:
                        raise ValueError(
                            f"Image {path} has only {src.count} bands, but at least 55 bands are required. "
                            f"This dataloader is configured to use specific bands (9, 18, 32, 42, 55)."
                        )

                    # Check data range for required bands
                    data = src.read(self.REQUIRED_BANDS)
                    valid_mask = ~np.isnan(data)
                    if np.any(valid_mask):  # Only check if there are valid values
                        valid_data = data[valid_mask]
                        # Warn if the image has values outside [-1, 1], but don't raise error — normalize_channel() will handle it.
                        if valid_data.min() < -1 or valid_data.max() > 1:
                            logger.warning(
                                f"Image {path} has out-of-bound data range [{valid_data.min()}, {valid_data.max()}] "
                                f"before normalization. This will be corrected by normalize_channel()."
                            )
            except rasterio.errors.RasterioIOError as e:
                raise ValueError(f"Failed to open image {path}: {str(e)}")
            except Exception as e:
                raise ValueError(f"Unexpected error validating {path}: {str(e)}")

    # computing per-band min/max from the full band, not just from foreground-only pixels
    def normalize_channel(self, channel_data: np.ndarray) -> np.ndarray:
        """
        Per-channel normalization to [-1, 1] range for VAE compatibility.
        Handles NaN values as background.

        Args:
            channel_data: Input channel data, may contain NaN values for background

        Returns:
            Normalized channel data in [-1, 1] range, with NaN values preserved

        Implementation Notes:
        -------------------
        1. Normalization Strategy:
           - Two-step normalization: [0,1] then [-1,1]
           - Only normalizes valid (non-NaN) regions
           - Required for VAE training stability
           - Preserves spectral relationships
           - Normalization is done per-image, per-band, to avoid dataset-level leakage

        2. Background Handling:
           - NaN values represent background
           - Preserves NaN values for accurate mask generation; background is not learned by the model.
           - No background inpainting
           - Focuses on leaf features
        """
        # Create a mask for non-NaN values (leaf regions)
        valid_mask = ~np.isnan(channel_data)

        # Log per-channel NaN and stats before normalization
        # if np.isnan(channel_data).any():
        #     logger.warning(f"[Normalize] NaNs found before normalization — min: {np.nanmin(channel_data):.4f}, max: {np.nanmax(channel_data):.4f}, mean: {np.nanmean(channel_data):.4f}")

        if not np.any(valid_mask):
            logger.warning("Channel contains only NaN values (background). Returning NaN array.")
            return np.full_like(channel_data, np.nan, dtype=np.float32)

        # Calculate min/max only on valid values (leaf regions)
        min_val = np.nanmin(channel_data)
        max_val = np.nanmax(channel_data)

        # Safety check for division by zero
        if max_val == min_val:
            logger.warning(f"Channel has constant value {min_val} in leaf regions. Returning zero array.")
            return np.zeros_like(channel_data, dtype=np.float32)

        # Normalize to [0, 1] then scale to [-1, 1]
        normalized = np.full_like(channel_data, np.nan, dtype=np.float32)
        normalized[valid_mask] = (channel_data[valid_mask] - min_val) / (max_val - min_val)
        normalized[valid_mask] = 2 * normalized[valid_mask] - 1

        return normalized

        # Modified padding logic to use the same per-band mean value as the normalization baseline, 
        # ensuring that padded background areas resemble the distribution of valid leaf surroundings
        # 2.7.: bounding box artifact persists. Filling with mean should avoid bounding box artifacts as long as per_band_means is 
        # computed only on foreground (leaf) pixels! BUT image contains very little leaf area, so maybe the mean may not be suitable
    def pad_to_square(self, img: torch.Tensor, fill_value: float = None) -> torch.Tensor:
        """
        Pad a (C, H, W) tensor to a square shape (C, S, S) with the given fill value.
        The fill_value should be computed from valid foreground pixels to prevent artificial edges.
        If fill_value is None, raises an error.

        Design Note:
        ------------
        Using the mean of valid (foreground) pixels as the fill value is a deliberate design choice to minimize 
        artificial edges at the image border. This prevents the VAE decoder from learning spurious "bounding box" artifacts 
        and ensures that padded regions are statistically similar to the leaf regions, supporting robust scientific analysis.
        """
        if fill_value is None:
            raise ValueError("Fill value for padding must be explicitly set based on foreground data.")
        c, h, w = img.shape
        size = max(h, w, self.resolution)
        pad_h = size - h
        pad_w = size - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        img = torch.nn.functional.pad(img, padding, value=fill_value)
        return img

    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Load and preprocess a multispectral image.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (preprocessed image tensor, background mask)
            - Image tensor shape: (5, resolution, resolution)
            - Mask tensor shape: (1, resolution, resolution) if return_mask=True

        Implementation Notes:
        -------------------
        1. Image Processing Pipeline:
           - Band selection and extraction
           - Per-channel normalization
           - Fixed resolution resizing
           - Tensor conversion
           - Background mask generation

        2. Background Handling:
           - NaN values represent background
           - Generates binary mask (1 for leaf, 0 for background)
           - Background (cut-out or padding) excluded from training entirely; loss is masked to focus on leaf pixels only.
           - Optional mask return for loss computation

        3. Data Flow:
           - NaNs are preserved until after normalization for correct mask generation.
           - Background NaNs are filled with the per-band mean to avoid introducing out-of-distribution values, which would destabilize the VAE and bias the loss.
           - The mask is generated from the first band, assuming all bands have the same NaN pattern (safe for this dataset).
           - Padding is performed before resizing to preserve aspect ratio.
           - Resizing is performed after padding to ensure the final tensor matches the model's expected input size.
           - Mask is resized with nearest-neighbor interpolation to preserve binary values (1=leaf, 0=background).
        """
        try:
            with rasterio.open(image_path) as src:
                # Read required bands
                image = src.read(self.REQUIRED_BANDS)  # Shape: (5, height, width)

                # Log raw band values for NaN and stats
                # if np.isnan(image).any():
                #     logger.warning(f"[Preprocess] NaNs detected in raw image BEFORE normalization — shape: {image.shape}, min: {np.nanmin(image):.4f}, max: {np.nanmax(image):.4f}, mean: {np.nanmean(image):.4f}")

                # Generate background mask from NaN values
                # Use first band to create mask (all bands should have same NaN pattern)
                background_mask = np.isnan(image[0]).astype(np.float32)
                leaf_mask = 1 - background_mask  # 1 for leaf, 0 for background

                # Convert to float32 and normalize
                image = image.astype(np.float32)
                normalized_image = np.zeros_like(image)
                # fill NaN with the mean value of each band (computed from the valid pixels in that image)
                # This step ensures that background values are within the natural data distribution, minimizing decoder confusion and potential artifacts.
                for i in range(5):
                    band = image[i]
                    nan_mask = np.isnan(band)
                    mean_val = np.nanmean(band)
                    band[nan_mask] = mean_val
                    normalized_image[i] = self.normalize_channel(band)

                # Convert to tensor, resizing is performed after padding]
                image_tensor = torch.from_numpy(normalized_image)
                if torch.isnan(image_tensor).any():
                    logger.info(f"[Sanitize] Replacing NaNs in input tensor with 0.0 to avoid propagation into model.")
                    image_tensor = torch.nan_to_num(image_tensor, nan=0.0)

                # Compute fill_value as mean of valid (foreground) pixels per band for padding
                # This prevents artificial edges by matching the padded value to the leaf distribution
                # (fixes "bounding box" style artifacts)
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
                image_tensor = self.pad_to_square(image_tensor, fill_value=fill_value)
                # Now resize to (resolution, resolution) if needed (should be square already)
                if image_tensor.shape[1] != self.resolution or image_tensor.shape[2] != self.resolution:
                    image_tensor = F.interpolate(
                        image_tensor.unsqueeze(0),
                        size=(self.resolution, self.resolution),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)

                # Inspect tensor for NaNs before interpolation
                if torch.isnan(image_tensor).any():
                    logger.warning(f"[ToTensor] NaNs detected AFTER conversion to tensor — min: {image_tensor.min().item():.4f}, max: {image_tensor.max().item():.4f}, mean: {image_tensor.mean().item():.4f}")

                # Pad and resize mask
                mask_tensor = torch.from_numpy(leaf_mask).unsqueeze(0)  # (1, H, W)
                mask_tensor = self.pad_to_square(mask_tensor, fill_value=0.0)
                if mask_tensor.shape[1] != self.resolution or mask_tensor.shape[2] != self.resolution:
                    mask_tensor = F.interpolate(
                        mask_tensor.unsqueeze(0),
                        size=(self.resolution, self.resolution),
                        mode='nearest'
                    ).squeeze(0)

                # All non-leaf (background) regions are zeroed out after resizing.
                # This avoids any contribution from padding or masked areas during training, as background has zero importance.
                # Explicitly zero out background regions (non-leaf) after resizing
                image_tensor = image_tensor * mask_tensor

                if self.return_mask:
                    return image_tensor, mask_tensor
                return image_tensor

        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a preprocessed image and optional background mask.

        Args:
            idx: Index of the image to get

        Returns:
            If return_mask=True:
                Tuple of (preprocessed image tensor, background mask)
            Otherwise:
                Preprocessed image tensor

        Implementation Notes:
        -------------------
        1. Caching Strategy:
           - Checks cache before processing
           - Stores processed images and masks
           - Reduces computation overhead (especially important for large datasets)

        2. Transform Pipeline:
           - Applies additional transforms if specified
           - Must be NaN-safe
           - Maintains data consistency
           - Supports augmentation (applied only to the image, not the mask)
        """
        image_path = str(self.image_paths[idx])

        # Check cache first
        if self.use_cache and image_path in self.cache:
            return self.cache[image_path]

        # Load and preprocess image
        if self.return_mask:
            image_tensor, mask_tensor = self.preprocess_image(image_path)
        else:
            image_tensor = self.preprocess_image(image_path)

        # Apply additional transforms if specified
        if self.transform:
            if self.return_mask:
                # Apply transforms to image only, not mask
                image_tensor = self.transform(image_tensor)
            else:
                image_tensor = self.transform(image_tensor)

        # Cache the result if caching is enabled
        # Caching is critical for large datasets to avoid repeated disk I/O and speed up training.
        if self.use_cache:
            if self.return_mask:
                self.cache[image_path] = (image_tensor, mask_tensor)
            else:
                self.cache[image_path] = image_tensor

        if self.return_mask:
            return image_tensor, mask_tensor
        return image_tensor

def create_vae_dataloaders(
    train_list_path: str,
    val_list_path: str,
    batch_size: int = 4,
    resolution: int = 512,
    num_workers: int = 4,
    use_cache: bool = True,
    prefetch_factor: Optional[int] = 2,
    persistent_workers: bool = True,
    return_mask: bool = True  # New parameter to control mask return
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders for VAE training.

    Args:
        train_list_path: Path to train_files.txt
        val_list_path: Path to val_files.txt
        batch_size: Batch size for training
        resolution: Target resolution for images
        num_workers: Number of worker processes
        use_cache: Whether to cache loaded images
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Whether to keep workers alive between epochs
        return_mask: Whether to return background masks

    Returns:
        Tuple of (train_loader, val_loader)

    Implementation Notes:
    -------------------
    1. Factory Function Design:
       - Creates both train and val dataloaders
       - Ensures consistent configuration
       - Optimizes for GPU training
       - Handles background masking (background is excluded from training)

    2. DataLoader Configuration:
       - pin_memory=True for faster GPU transfer (critical for large-scale VAE training)
       - persistent_workers for efficient process management
       - prefetch_factor for optimized loading
       - drop_last=True to avoid partial batches (important for batchnorm and reproducibility)

    3. Train/Val Separation:
       - Training data is shuffled
       - Validation data is not shuffled
       - Maintains proper data separation
    """
    # Create datasets
    train_dataset = VAEMultispectralDataset(
        file_list_path=train_list_path,
        resolution=resolution,
        use_cache=use_cache,
        return_mask=return_mask
    )

    val_dataset = VAEMultispectralDataset(
        file_list_path=val_list_path,
        resolution=resolution,
        use_cache=use_cache,
        return_mask=return_mask
    )
    
    # Common DataLoader kwargs
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,  # Faster GPU transfer
        "persistent_workers": persistent_workers and num_workers > 0,  # Efficient worker management
        "drop_last": True  # Avoid partial batches
    }
    
    # Add prefetch_factor if specified and num_workers > 0
    if prefetch_factor is not None and num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    
    # Create dataloaders with appropriate settings
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,  # Shuffle training data
        **loader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,  # Don't shuffle validation data
        **loader_kwargs
    )
    
    return train_loader, val_loader 
