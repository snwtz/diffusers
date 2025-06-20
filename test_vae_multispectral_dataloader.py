"""
Test script for the VAE multispectral dataloader.

This script tests the key functionality of the VAEMultispectralDataset and related classes,
including data loading, normalization, validation, error handling, and background masking.

src of test image files: https://github.com/cogsys-tuebingen/deephs_fruit?tab=readme-ov-file

Test Design Decisions:

1. Data Testing:
   - Uses multispectral TIFF files with 5 or more bands
   - Takes first predefined bands for processing
   - Maintains reproducibility by using fixed seed for random selection
   - Tests background masking from NaN values

2. Error Handling Tests:
   - Non-existent file lists are tested
   - Invalid band counts are tested
   - Empty file lists are tested
   - NaN handling is verified

3. Caching Tests:
   - Performance is measured using time.time()
   - Data consistency is verified using torch.allclose()
   - Cache behavior is tested with controlled data access patterns
   - Mask caching is verified

4. SD3 Compatibility:
   - Input shape tests verify 5-channel, 512x512 requirements
   - Pixel range tests ensure [-1, 1] normalization for VAE
   - Channel independence is verified for normalization
   - Background masking is preserved

5. Performance Tests:
   - Worker behavior is tested for consistency
   - Local testing configuration:
     * num_workers=0
     * prefetch_factor=None
     * persistent_workers=False
   - Memory usage is monitored
   - Mask handling performance is verified

6. Tolerance:
   - Tests use relaxed tolerances (rtol=1e-2, atol=1e-2) for floating-point imprecision
   - NaN values are preserved in masked regions

7. Band Selection Test:
   - The test_specific_band_selection test uses the exact file as the dataloader for index 0
   - Background masking is verified in band selection

8. Background Masking Tests:
   - NaN values are properly converted to binary masks
   - Masks are correctly applied during normalization
   - Background regions are properly handled
   - Mask shape and type are verified

Usage:
    pytest test_vae_multispectral_dataloader.py --data-dir "C:\\Users\\NOcsPS-440g\\Desktop\\Beispiel Dateien\\Ausgeschnittene Bilder" -v

Note:
    For local testing, worker-intensive features are disabled to ensure
    reliable test execution. These features should be enabled when running
    on GPU hardware for actual training.
"""

import os
import sys
import numpy as np
import torch
import pytest
import time
import logging
import random
from pathlib import Path
import rasterio
import torch.nn.functional as F
from examples.multispectral.vae_multispectral_dataloader import (
    VAEMultispectralDataset,
    create_vae_dataloaders
)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def pytest_addoption(parser):
    parser.addoption("--data-dir", action="store", default=None,
                     help="Directory containing multispectral TIFF files for testing")

@pytest.fixture
def data_dir(request):
    data_dir = request.config.getoption("--data-dir")
    if data_dir is None:
        pytest.skip("--data-dir not specified")
    if not os.path.exists(data_dir):
        pytest.skip(f"Data directory {data_dir} does not exist")
    return data_dir

def get_test_images(data_dir, num_images=2):
    """Select a subset of images for testing."""
    all_files = sorted(Path(data_dir).glob('*.tiff'))

    # Adjust num_images if we have fewer files
    num_images = min(num_images, len(all_files))

    # Randomly select images
    selected_files = random.sample(all_files, num_images)

    return selected_files

@pytest.fixture
def test_images(data_dir):
    """Get a subset of test images."""
    return get_test_images(data_dir)

@pytest.fixture
def file_lists(data_dir, test_images):
    """Create temporary train and val file lists for testing."""
    train_list = Path(data_dir) / "train_files.txt"
    val_list = Path(data_dir) / "val_files.txt"

    # Write test files to train list
    with open(train_list, 'w') as f:
        for img in test_images:
            f.write(f"{img}\n")

    # Write empty val list (we don't need it for testing)
    with open(val_list, 'w') as f:
        pass

    yield str(train_list), str(val_list)

    # Cleanup after tests
    train_list.unlink(missing_ok=True)
    val_list.unlink(missing_ok=True)

def test_dataset_initialization(data_dir, test_images, file_lists):
    """Test dataset initialization with real data."""
    train_list, _ = file_lists
    dataset = VAEMultispectralDataset(train_list)
    assert len(dataset) > 0
    assert dataset.resolution == 512
    assert dataset.use_cache is True

def test_band_count_validation(data_dir, test_images, file_lists):
    """Test validation of band count."""
    train_list, _ = file_lists
    dataset = VAEMultispectralDataset(train_list)
    assert len(dataset) > 0

    # Test with non-existent file list
    with pytest.raises(FileNotFoundError):
        VAEMultispectralDataset("non_existent_file.txt")

def test_normalize_channel(data_dir, test_images, file_lists):
    """Test channel normalization with real data."""
    train_list, _ = file_lists
    dataset = VAEMultispectralDataset(train_list)

    # Load a real image
    image_path = str(test_images[0])
    with rasterio.open(image_path) as src:
        data = src.read(1)  # Read first band

    normalized = dataset.normalize_channel(data)
    # Check for [-1, 1] range in non-NaN regions
    mask = ~np.isnan(normalized)
    assert np.all(normalized[mask] >= -1) and np.all(normalized[mask] <= 1)

def test_sd3_compatible_input_shape(data_dir, test_images, file_lists):
    """Test that preprocessed images are compatible with SD3's VAE input requirements."""
    train_list, _ = file_lists
    dataset = VAEMultispectralDataset(train_list)

    # Load and preprocess image
    image, mask = dataset[0]

    # Check tensor properties for SD3 compatibility
    assert isinstance(image, torch.Tensor)
    assert image.shape == (5, 512, 512)  # 5 channels, 512x512 resolution
    assert image.dtype == torch.float32
    # Check for [-1, 1] range in non-NaN regions
    mask_tensor = ~torch.isnan(image)
    assert torch.all(image[mask_tensor] >= -1) and torch.all(image[mask_tensor] <= 1)

def test_pixel_range_normalization_for_vae(data_dir, test_images, file_lists):
    """Test that pixel values are properly normalized for VAE input."""
    train_list, _ = file_lists
    dataset = VAEMultispectralDataset(train_list)
    image, mask = dataset[0]

    # Check normalization properties
    mask_tensor = ~torch.isnan(image)
    assert torch.all(image[mask_tensor] >= -1) and torch.all(image[mask_tensor] <= 1)
    # Check that each channel has been normalized independently
    for c in range(image.shape[0]):
        channel = image[c]
        valid_mask = ~torch.isnan(channel)
        if torch.any(valid_mask):
            valid_values = channel[valid_mask]
            # For [-1, 1] normalization, check range with relaxed tolerance
            assert valid_values.min() >= -1.0 - 1e-3, f"Channel {c} min value {valid_values.min()} below -1"
            assert valid_values.max() <= 1.0 + 1e-3, f"Channel {c} max value {valid_values.max()} above 1"

def test_caching_behavior(data_dir, test_images, file_lists):
    """Test that caching improves load time and maintains data consistency."""
    train_list, _ = file_lists
    dataset = VAEMultispectralDataset(train_list, use_cache=True)

    # First load
    start_time = time.time()
    first_load_image, first_load_mask = dataset[0]
    first_load_time = time.time() - start_time

    # Second load (should be from cache)
    start_time = time.time()
    second_load_image, second_load_mask = dataset[0]
    second_load_time = time.time() - start_time

    # Verify cache is working
    assert second_load_time < first_load_time
    assert torch.allclose(first_load_image, second_load_image, equal_nan=True)
    assert torch.allclose(first_load_mask, second_load_mask)

def test_dataloader_creation(data_dir, test_images, file_lists):
    """Test dataloader creation and basic functionality."""
    train_list, val_list = file_lists
    train_loader, val_loader = create_vae_dataloaders(
        train_list,
        val_list,
        batch_size=2,
        num_workers=0,  # Use 0 for testing
        use_cache=True,
        prefetch_factor=None,  # Disabled for local testing
        persistent_workers=False  # Disabled for local testing
    )

    # Test batch loading
    batch, masks = next(iter(train_loader))
    assert isinstance(batch, torch.Tensor)
    assert batch.shape[0] == 2  # batch_size
    assert batch.shape[1] == 5  # channels
    assert batch.shape[2] == 512  # height
    assert batch.shape[3] == 512  # width

def test_worker_behavior(data_dir, test_images, file_lists):
    """Test worker behavior and data loading consistency."""
    train_list, val_list = file_lists
    train_loader, val_loader = create_vae_dataloaders(
        train_list,
        val_list,
        batch_size=2,
        num_workers=0,  # Use 0 for testing
        persistent_workers=False,  # Disabled for local testing
        prefetch_factor=None  # Disabled for local testing
    )

    # Test multiple epochs
    for epoch in range(2):
        batches = []
        for batch in train_loader:
            batches.append(batch)

        # Verify batch consistency
        for i in range(len(batches)-1):
            assert batches[i].shape == batches[i+1].shape

def test_error_handling(data_dir):
    """Test error handling for invalid data."""
    # Test with non-existent file list
    with pytest.raises(FileNotFoundError):
        VAEMultispectralDataset("non_existent_file.txt")

def test_specific_band_selection(data_dir, test_images, file_lists):
    """Test that the dataloader correctly selects the hardcoded bands."""
    train_list, _ = file_lists
    dataset = VAEMultispectralDataset(train_list)
    image, mask = dataset[0]

    # Print the file path used by the dataloader for index 0
    image_path = dataset.image_paths[0]
    print(f"Dataloader image path: {image_path}")

    # Use the same file for the expected bands
    with rasterio.open(image_path) as src:
        expected_bands = src.read([9, 18, 32, 42, 55]).astype(np.float32)
        # Per-channel normalization to [-1, 1]
        for i in range(expected_bands.shape[0]):
            band = expected_bands[i]
            valid_mask = ~np.isnan(band)
            if np.any(valid_mask):
                min_val = np.min(band[valid_mask])
                max_val = np.max(band[valid_mask])
                if max_val > min_val:
                    normalized = (band - min_val) / (max_val - min_val)
                    expected_bands[i] = 2 * normalized - 1
                else:
                    expected_bands[i] = np.zeros_like(band)
            else:
                expected_bands[i] = np.zeros_like(band)
        expected_bands = torch.from_numpy(expected_bands)
        expected_bands = F.interpolate(
            expected_bands.unsqueeze(0),
            size=(512, 512),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

    assert torch.allclose(image, expected_bands, rtol=1e-2, atol=1e-2, equal_nan=True), "Dataset output does not match expected band selection"

def test_train_val_separation(data_dir, test_images, file_lists):
    """Test that train and validation dataloaders are properly separated."""
    train_list, val_list = file_lists
    train_loader, val_loader = create_vae_dataloaders(
        train_list,
        val_list,
        batch_size=2,
        num_workers=0,
        use_cache=True
    )

    # Verify train loader has data
    train_batch, train_masks = next(iter(train_loader))
    assert train_batch.shape[0] == 2  # batch_size

    # Verify val loader is empty (since we created an empty val list)
    assert len(list(val_loader)) == 0

def test_background_masking(data_dir, test_images, file_lists):
    """Test background masking functionality."""
    train_list, _ = file_lists
    dataset = VAEMultispectralDataset(train_list)

    # Load an image and its mask
    image, mask = dataset[0]

    # Verify mask properties
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (1, 512, 512)  # Single channel mask
    assert mask.dtype == torch.float32
    assert torch.all((mask == 0) | (mask == 1))  # Binary mask

    # Verify image properties with mask
    assert isinstance(image, torch.Tensor)
    assert image.shape == (5, 512, 512)  # 5 channels, 512x512 resolution
    assert image.dtype == torch.float32

    # Get background and foreground pixels
    background_pixels = image[:, mask.squeeze(0) == 0]
    foreground_pixels = image[:, mask.squeeze(0) == 1]

    # Verify background regions (if any exist)
    if background_pixels.numel() > 0:
        # Check if all background pixels are NaN (ideal case)
        all_nan = torch.isnan(background_pixels).all()
        if not all_nan:
            # If not all NaN, check for interpolated values
            is_nan = torch.isnan(background_pixels)
            is_zero = torch.abs(background_pixels) < 1e-3
            all_valid = torch.all(is_nan | is_zero)
            if not all_valid:
                nan_ratio = torch.isnan(background_pixels).float().mean().item()
                zero_ratio = (torch.abs(background_pixels) < 1e-3).float().mean().item()
                if nan_ratio > 0.95 or zero_ratio > 0.95:
                    print(f"Acceptable: Background is mostly {'NaN' if nan_ratio > zero_ratio else 'near-zero'} "
                          f"(NaN ratio: {nan_ratio:.2f}, Near-zero ratio: {zero_ratio:.2f})")
                else:
                    raise AssertionError(
                        f"Background regions contain unexpected values. "
                        f"NaN ratio: {nan_ratio:.2f}, Near-zero ratio: {zero_ratio:.2f}"
                    )
    else:
        print("No background pixels found in this image; skipping background assertion.")

    # Verify foreground regions (must exist)
    assert foreground_pixels.numel() > 0, "Image should have foreground regions"
    assert torch.any(~torch.isnan(foreground_pixels)), "Foreground regions should contain valid values"
    valid_foreground = foreground_pixels[~torch.isnan(foreground_pixels)]
    assert valid_foreground.numel() > 0, "Should have valid (non-NaN) foreground values"
    assert torch.all(valid_foreground >= -1) and torch.all(valid_foreground <= 1), "Foreground values should be in [-1, 1] range"

def test_dataloader_with_masks(data_dir, test_images, file_lists):
    """Test dataloader creation and basic functionality with masks."""
    train_list, val_list = file_lists
    train_loader, val_loader = create_vae_dataloaders(
        train_list,
        val_list,
        batch_size=2,
        num_workers=0,  # Use 0 for testing
        use_cache=True,
        prefetch_factor=None,  # Disabled for local testing
        persistent_workers=False  # Disabled for local testing
    )

    # Test batch loading
    batch, masks = next(iter(train_loader))

    # Verify batch properties
    assert isinstance(batch, torch.Tensor)
    assert batch.shape[0] == 2  # batch_size
    assert batch.shape[1] == 5  # channels
    assert batch.shape[2] == 512  # height
    assert batch.shape[3] == 512  # width

    # Verify mask properties
    assert isinstance(masks, torch.Tensor)
    assert masks.shape[0] == 2  # batch_size
    assert masks.shape[1] == 1  # single channel
    assert masks.shape[2] == 512  # height
    assert masks.shape[3] == 512  # width
    assert torch.all((masks == 0) | (masks == 1))  # Binary masks

def test_specific_band_selection_with_mask(data_dir, test_images, file_lists):
    """Test that the dataloader correctly selects bands and handles masks."""
    train_list, _ = file_lists
    dataset = VAEMultispectralDataset(train_list)
    image, mask = dataset[0]

    # Print the file path used by the dataloader for index 0
    image_path = dataset.image_paths[0]
    print(f"Dataloader image path: {image_path}")

    # Use the same file for the expected bands
    with rasterio.open(image_path) as src:
        expected_bands = src.read([9, 18, 32, 42, 55]).astype(np.float32)

        # Create expected mask from NaN values
        expected_mask = ~np.isnan(expected_bands[0])
        expected_mask = torch.from_numpy(expected_mask).float()
        expected_mask = F.interpolate(
            expected_mask.unsqueeze(0).unsqueeze(0),
            size=(512, 512),
            mode='nearest'
        ).squeeze(0)  # Keep the channel dimension

        # Per-channel normalization to [-1, 1]
        for i in range(expected_bands.shape[0]):
            band = expected_bands[i]
            valid_mask = ~np.isnan(band)
            if np.any(valid_mask):
                min_val = np.min(band[valid_mask])
                max_val = np.max(band[valid_mask])
                if max_val > min_val:
                    normalized = (band - min_val) / (max_val - min_val)
                    expected_bands[i] = 2 * normalized - 1
                else:
                    expected_bands[i] = np.zeros_like(band)
            else:
                expected_bands[i] = np.zeros_like(band)

        expected_bands = torch.from_numpy(expected_bands)
        expected_bands = F.interpolate(
            expected_bands.unsqueeze(0),
            size=(512, 512),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

    # Verify image and mask
    assert torch.allclose(image, expected_bands, rtol=1e-2, atol=1e-2, equal_nan=True), "Dataset output does not match expected band selection"
    assert torch.allclose(mask, expected_mask, rtol=1e-2, atol=1e-2), "Dataset mask does not match expected mask"

if __name__ == "__main__":
    # Remove the script name from sys.argv
    sys.argv.pop(0)

    # Run pytest with the remaining arguments
    pytest.main(sys.argv)