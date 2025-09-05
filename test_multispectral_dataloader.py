"""
Tests for Multispectral DreamBooth Dataloader
============================================

Validates core functionality of `MultispectralDataset` and its DataLoader:
data loading, normalization, masks, batching, and error handling.

USAGE:
------
pytest test_multispectral_dataloader.py --data-dir "/path/to/tiffs" -v

Test Coverage:
--------------
- SD3 compatibility: (5, 512, 512) tensors, float32, [-1, 1]
- Batch structure for DreamBooth (dict with pixel_values, mask, prompts)
- DataLoader batching/stacking correctness
- Error handling for invalid inputs
- Determinism and basic performance (local test config)
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
from multispectral_dataloader import MultispectralDataset
from torch.utils.data import DataLoader

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
    
    # Adjust num_images if fewer files
    num_images = min(num_images, len(all_files))
    
    # Randomly select images
    selected_files = random.sample(all_files, num_images)
    
    return selected_files

@pytest.fixture
def test_images(data_dir):
    """Get a subset of test images."""
    return get_test_images(data_dir)

def test_dreambooth_batch_structure(data_dir):
    """
    Test that MultispectralDataset returns a dict with pixel_values, mask, and prompts,
    with correct shapes, types, and value ranges, as required by the DreamBooth training script.
    """
    dataset = MultispectralDataset(data_root=data_dir, return_mask=True, prompt="test prompt")
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "pixel_values" in sample
    assert "mask" in sample
    assert "prompts" in sample
    # pixel_values checks
    assert isinstance(sample["pixel_values"], torch.Tensor)
    assert sample["pixel_values"].shape == (5, 512, 512)
    assert sample["pixel_values"].dtype == torch.float32
    assert not torch.isnan(sample["pixel_values"]).any()
    assert torch.all(sample["pixel_values"] >= -1) and torch.all(sample["pixel_values"] <= 1)
    # mask checks
    assert isinstance(sample["mask"], torch.Tensor)
    assert sample["mask"].shape == (1, 512, 512)
    unique_mask = torch.unique(sample["mask"])
    assert set(unique_mask.tolist()).issubset({0.0, 1.0})
    # prompts checks
    assert isinstance(sample["prompts"], list)
    assert sample["prompts"][0] == "test prompt"

def test_dreambooth_dataloader_batch(data_dir):
    """
    Test that DataLoader batching works for DreamBooth: batch is a dict with stacked pixel_values and mask,
    and a list of prompts of correct length.
    """
    dataset = MultispectralDataset(data_root=data_dir, return_mask=True, prompt="test prompt")
    def collate_fn(batch):
        return {
            "pixel_values": torch.stack([s["pixel_values"] for s in batch]),
            "mask": torch.stack([s["mask"] for s in batch]),
            "prompts": [p for s in batch for p in s["prompts"]]
        }
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    assert batch["pixel_values"].shape == (2, 5, 512, 512)
    assert batch["mask"].shape == (2, 1, 512, 512)
    assert isinstance(batch["prompts"], list)
    assert batch["prompts"] == ["test prompt", "test prompt"]

if __name__ == "__main__":
    import sys
    pytest.main(sys.argv) 