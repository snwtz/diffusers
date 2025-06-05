#!/usr/bin/env python3
"""
Environment Setup Test Script

This script verifies that all required dependencies and configurations
are properly set up for the multispectral VAE training pipeline.
"""

import os
import sys
import torch
import rasterio
import numpy as np
from pathlib import Path
import wandb
from PIL import Image

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    try:
        import diffusers
        import transformers
        import accelerate
        import datasets
        print("✅ All Python packages imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    return True

def test_cuda():
    """Test CUDA availability and memory."""
    print("\nTesting CUDA...")
    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return False
    
    print(f"✅ CUDA is available")
    print(f"✅ CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return True

def test_rasterio():
    """Test rasterio installation and GDAL."""
    print("\nTesting rasterio...")
    try:
        print(f"✅ Rasterio version: {rasterio.__version__}")
        print(f"✅ GDAL version: {rasterio.__gdal_version__}")
    except Exception as e:
        print(f"❌ Rasterio error: {e}")
        return False
    return True

def test_wandb():
    """Test wandb login."""
    print("\nTesting wandb...")
    try:
        wandb.login()
        print("✅ Wandb login successful")
    except Exception as e:
        print(f"❌ Wandb error: {e}")
        return False
    return True

def test_directory_structure():
    """Test required directory structure."""
    print("\nTesting directory structure...")
    required_dirs = [
        "examples/multispectral",
        "configs",
        "output",
        "data/raw",
        "data/processed"
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Missing directory: {dir_path}")
            return False
    
    print("✅ All required directories exist")
    return True

def test_file_permissions():
    """Test file permissions."""
    print("\nTesting file permissions...")
    test_dirs = ["output", "data/raw", "data/processed"]
    
    for dir_path in test_dirs:
        test_file = Path(dir_path) / "test.txt"
        try:
            test_file.write_text("test")
            test_file.unlink()
            print(f"✅ Write permissions in {dir_path}")
        except Exception as e:
            print(f"❌ Permission error in {dir_path}: {e}")
            return False
    
    return True

def main():
    """Run all tests."""
    print("Starting environment setup tests...\n")
    
    tests = [
        ("Python imports", test_imports),
        ("CUDA setup", test_cuda),
        ("Rasterio setup", test_rasterio),
        ("Wandb setup", test_wandb),
        ("Directory structure", test_directory_structure),
        ("File permissions", test_file_permissions)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n=== Testing {test_name} ===")
        if not test_func():
            all_passed = False
            print(f"❌ {test_name} failed")
        else:
            print(f"✅ {test_name} passed")
    
    if all_passed:
        print("\n🎉 All tests passed! Environment is ready for training.")
    else:
        print("\n❌ Some tests failed. Please fix the issues before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main() 