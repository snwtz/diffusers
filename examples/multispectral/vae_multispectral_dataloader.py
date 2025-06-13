"""
Multispectral Image Dataloader for VAE Training

This module implements a specialized dataloader for multispectral TIFF images,
optimized for training a VAE on hyperspectral plant data. It extends the base
multispectral dataloader to support train/val splits via file lists.

Key Features:
- Support for train/val splits via file lists
- Optimized for VAE training
- Memory-efficient loading
- Spectral fidelity preservation
- Robust validation
"""

import os
import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from typing import Optional, List, Tuple
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
    
    2. Normalization:
       - Per-channel normalization to [-1, 1] range
       - Required for VAE training stability
       - Preserves spectral relationships
    
    3. Memory Management:
       - Optional caching for repeated access
       - Efficient worker process utilization
       - GPU memory considerations
t    """
    
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
        use_cache: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            file_list_path: Path to train_files.txt or val_files.txt
            resolution: Target resolution for images (default: 512)
            transform: Additional transforms to apply
            use_cache: Whether to cache loaded images in memory
        
        Implementation Notes:
        -------------------
        1. File List Handling:
           - Supports train/val splits via file lists
           - Validates all files on initialization
           - Ensures consistent data access
        
        2. Caching Strategy:
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
        self.cache = {} if use_cache else None
        
        # Validate all images on initialization
        self._validate_all_images()
        logger.info(f"Loaded {len(self.image_paths)} files from {file_list_path}")
    
    def _validate_all_images(self):
        """
        Validate that all images have at least 55 bands and correct data range.
        
        Implementation Notes:
        -------------------
        1. Band Count Validation:
           - Ensures all images have required bands
           - Prevents runtime errors during training
           - Maintains data consistency
        
        2. Data Range Validation:
           - Ensures data is properly normalized
           - Prevents training instability
           - Maintains VAE compatibility
        
        3. Error Handling:
           - Comprehensive error messages
           - Early failure for invalid data
           - Clear debugging information
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
                    if not (-1 <= data.min() <= data.max() <= 1):
                        raise ValueError(
                            f"Image {path} has invalid data range [{data.min()}, {data.max()}]. "
                            f"Data must be normalized to [-1, 1] range."
                        )
            except rasterio.errors.RasterioIOError as e:
                raise ValueError(f"Failed to open image {path}: {str(e)}")
            except Exception as e:
                raise ValueError(f"Unexpected error validating {path}: {str(e)}")
    
    def normalize_channel(self, channel_data: np.ndarray) -> np.ndarray:
        """
        Per-channel normalization to [-1, 1] range for VAE compatibility.
        
        Args:
            channel_data: Input channel data
            
        Returns:
            Normalized channel data in [-1, 1] range
            
        Implementation Notes:
        -------------------
        1. Normalization Strategy:
           - Two-step normalization: [0,1] then [-1,1]
           - Required for VAE training stability
           - Preserves spectral relationships
        
        2. Error Handling:
           - Handles NaN values
           - Prevents division by zero
           - Maintains data integrity
        """
        # Handle NaN values
        min_val = np.nanmin(channel_data)
        max_val = np.nanmax(channel_data)
        
        # Safety check for division by zero
        if max_val == min_val:
            logger.warning(f"Channel has constant value {min_val}. Returning zero array.")
            return np.zeros_like(channel_data, dtype=np.float32)
            
        # Normalize to [0, 1] then scale to [-1, 1]
        normalized = (channel_data - min_val) / (max_val - min_val)
        return 2 * normalized - 1
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess a multispectral image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor of shape (5, resolution, resolution)
            
        Implementation Notes:
        -------------------
        1. Image Processing Pipeline:
           - Band selection and extraction
           - Per-channel normalization
           - Fixed resolution resizing
           - Tensor conversion
        
        2. Memory Efficiency:
           - Minimal memory copies
           - Efficient tensor operations
           - Proper cleanup
        """
        try:
            with rasterio.open(image_path) as src:
                # Read required bands
                image = src.read(self.REQUIRED_BANDS)  # Shape: (5, height, width)
                
                # Convert to float32 and normalize
                image = image.astype(np.float32)
                normalized_image = np.zeros_like(image)
                for i in range(5):
                    normalized_image[i] = self.normalize_channel(image[i])
                
                # Convert to tensor and resize
                image_tensor = torch.from_numpy(normalized_image)
                image_tensor = F.interpolate(
                    image_tensor.unsqueeze(0),
                    size=(self.resolution, self.resolution),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                
                return image_tensor
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a preprocessed image.
        
        Args:
            idx: Index of the image to get
            
        Returns:
            Preprocessed image tensor
            
        Implementation Notes:
        -------------------
        1. Caching Strategy:
           - Checks cache before processing
           - Stores processed images
           - Reduces computation overhead
        
        2. Transform Pipeline:
           - Applies additional transforms if specified
           - Maintains data consistency
           - Supports augmentation
        """
        image_path = str(self.image_paths[idx])
        
        # Check cache first
        if self.use_cache and image_path in self.cache:
            return self.cache[image_path]
        
        # Load and preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Apply additional transforms if specified
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        # Cache the result if caching is enabled
        if self.use_cache:
            self.cache[image_path] = image_tensor
        
        return image_tensor

def create_vae_dataloaders(
    train_list_path: str,
    val_list_path: str,
    batch_size: int = 4,
    resolution: int = 512,
    num_workers: int = 4,
    use_cache: bool = True,
    prefetch_factor: Optional[int] = 2,
    persistent_workers: bool = True
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
    
    Returns:
        Tuple of (train_loader, val_loader)
        
    Implementation Notes:
    -------------------
    1. Factory Function Design:
       - Creates both train and val dataloaders
       - Ensures consistent configuration
       - Optimizes for GPU training
    
    2. DataLoader Configuration:
       - pin_memory=True for faster GPU transfer
       - persistent_workers for efficient process management
       - prefetch_factor for optimized loading
       - drop_last=True to avoid partial batches
    
    3. Train/Val Separation:
       - Training data is shuffled
       - Validation data is not shuffled
       - Maintains proper data separation
    """
    # Create datasets
    train_dataset = VAEMultispectralDataset(
        file_list_path=train_list_path,
        resolution=resolution,
        use_cache=use_cache
    )
    
    val_dataset = VAEMultispectralDataset(
        file_list_path=val_list_path,
        resolution=resolution,
        use_cache=use_cache
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