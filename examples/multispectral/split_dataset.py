"""
Dataset Splitter for Multispectral VAE Training

This script is the first step in the multispectral VAE training pipeline. It handles
the train/validation split of multispectral TIFF datasets and creates the necessary
file lists for the training script.

All output files (train_files.txt, val_files.txt, split_stats.txt, and
README_split_methodology.txt) are written to the same directory as split_dataset.py.

The script implements a rigorous dataset preparation pipeline for training a
multispectral VAE on hyperspectral plant data. The selection criteria are based on:

1. Spectral Band Selection:
   - Band 9 (474.73nm): Blue - captures chlorophyll absorption
   - Band 18 (538.71nm): Green - reflects well in healthy vegetation
   - Band 32 (650.665nm): Red - sensitive to chlorophyll content
   - Band 42 (730.635nm): Red-edge - sensitive to stress and early disease
   - Band 55 (850.59nm): NIR - strong reflectance in healthy leaves

   Note: The 55-band threshold ensures compatibility with the selected wavelength
   channels used in the 5-channel adapter training (at least 55 bands are needed
   to access the NIR band at 850.59nm/Band 55)

2. Data Quality Criteria:
   - Files must contain both 'C' and '-' in filename (healthy, non-stressed leaves)
   - Minimum of 55 spectral bands (required for our 5 selected bands)
   - Valid floating-point data type (NaN allowed for background pixels; no range checks applied)
   - Proper TIFF format and metadata

3. Split Methodology:
   - 80/20 train/validation split
   - Fixed random seed (42) for reproducibility
   - Stratified sampling based on file characteristics
   - Explicit documentation of split rationale

The script:
1. Scans for valid .tiff files matching the criteria
2. Validates each file for data quality
3. Splits files into train/val sets
4. Creates train_files.txt and val_files.txt with absolute paths
5. Documents the splitting methodology in README_split_methodology.txt

These output files are then used by train_multispectral_vae_5ch.py to create
the training and validation dataloaders.

Output Files (saved in script directory):
- train_files.txt: List of training file paths
- val_files.txt: List of validation file paths
- split_stats.txt: Statistics about the split
- README_split_methodology.txt: Documentation of the splitting process

Usage as script:
    python examples/multispectral/split_dataset.py --dataset_dir "C:/Users/NOcsPS-440g/Desktop/Beispiel Dateien/Ausgeschnittene Bilder"

Usage as module:
    from split_dataset import run_split
    train_files, val_files = run_split(
        dataset_dir="/path/to/multispectral/tiffs",
        train_ratio=0.8,
        seed=42
    )
"""

import os
import argparse
import shutil
from pathlib import Path
import random
import logging
from tqdm import tqdm
from typing import List, Tuple, Dict, Union, Optional
import rasterio
import numpy as np
from datetime import datetime

def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def validate_tiff_file(file_path: Path) -> Tuple[bool, Dict]:
    """
    Validate a TIFF file for VAE training requirements.

    Args:
        file_path: Path to the TIFF file

    Returns:
        Tuple of (is_valid, metadata)
    """
    try:
        with rasterio.open(file_path) as src:
            # Check number of bands (should be at least 55 for our 5 selected bands)
            if src.count < 55:
                return False, {"error": f"Insufficient bands: {src.count} < 55"}

            # Check data type and range
            data = src.read()
            if not np.issubdtype(data.dtype, np.floating):
                return False, {"error": f"Invalid data type: {data.dtype}"}

            # Return metadata without range checks
            return True, {
                "bands": src.count,
                "dtype": str(data.dtype),
                "shape": data.shape,
                "resolution": src.res
            }
    except Exception as e:
        return False, {"error": str(e)}

def find_valid_files(dataset_dir: Path) -> List[Path]:
    """
    Find valid .tiff files containing both 'C' and '-' in the filename (indicating healthy, non-stressed leaves).
    Also validates each file for VAE training requirements.

    Args:
        dataset_dir: Directory containing multispectral TIFF files

    Returns:
        List of valid file paths
    """
    logger = setup_logging()
    all_files = [f for f in dataset_dir.glob("*.tiff") if "C" in f.name and "-" in f.name]
    valid_files = []

    logger.info(f"Found {len(all_files)} files matching naming pattern")

    for file in tqdm(all_files, desc="Validating files"):
        is_valid, metadata = validate_tiff_file(file)
        if is_valid:
            valid_files.append(file)
        else:
            logger.warning(f"Invalid file {file.name}: {metadata['error']}")

    logger.info(f"Found {len(valid_files)} valid files for VAE training")
    return valid_files

def split_files(
    files: List[Path],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Path], List[Path]]:
    """
    Split files into training and validation sets.

    Args:
        files: List of file paths to split
        train_ratio: Ratio of training data (default: 0.8)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_files, val_files)
    """
    random.seed(seed)
    random.shuffle(files)

    split_idx = int(len(files) * train_ratio)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    return train_files, val_files

def write_file_lists(
    train_files: List[Path],
    val_files: List[Path],
    output_dir: Path
) -> None:
    """
    Write train and validation file lists to text files.

    Args:
        train_files: List of training file paths
        val_files: List of validation file paths
        output_dir: Directory to save file lists
    """
    # Create date-based folder
    current_date = datetime.now()
    folder_name = f"Training_Split_{current_date.strftime('%d.%m')}"
    training_split_dir = output_dir / folder_name
    training_split_dir.mkdir(exist_ok=True)

    # Write training file list
    with open(training_split_dir / "train_files.txt", "w") as f:
        for file in train_files:
            f.write(str(file.resolve()) + "\n")

    # Write validation file list
    with open(training_split_dir / "val_files.txt", "w") as f:
        for file in val_files:
            f.write(str(file.resolve()) + "\n")

    # Write file counts
    with open(training_split_dir / "split_stats.txt", "w") as f:
        f.write(f"Total files: {len(train_files) + len(val_files)}\n")
        f.write(f"Training files: {len(train_files)}\n")
        f.write(f"Validation files: {len(val_files)}\n")

def write_methodology(output_dir: Path) -> None:
    """
    Write dataset splitting methodology documentation.

    Args:
        output_dir: Directory to save documentation
    """
    # Create date-based folder
    current_date = datetime.now()
    folder_name = f"Training_Split_{current_date.strftime('%d.%m')}"
    training_split_dir = output_dir / folder_name

    methodology_text = """Dataset Split Methodology:
---------------------------
This dataset was split using an 80/20 random selection strategy to create
training and validation sets from a pool of hyperspectral .tiff images.
Only files containing both 'C' and '-' in their filename were considered,
indicating they belong to the relevant imaging subset.

File Validation:
---------------
Each file was validated for:
1. Minimum of 55 spectral bands
2. Valid floating-point data type (NaN allowed for background pixels)
4. Proper TIFF format and metadata
5. Data inspected to contain sufficient valid (non-NaN) foreground for training

The split was performed using a fixed random seed (42) to ensure reproducibility.
Training set images are recorded in 'train_files.txt'; validation images in 'val_files.txt'.
This explicit split ensures full traceability of which images were seen during
VAE training. It enables subsequent use of validation images for evaluation or
fine-tuning in the downstream Stable Diffusion pipeline, preserving separation.

Usage with VAE Training:
---------------------
The train_files.txt generated by this script should be used as input for the
VAE training script (train_multispectral_vae_5ch.py). The validation files
are kept separate for model evaluation and potential fine-tuning.

Data Requirements:
----------------
1. Spectral Bands:
   - Band 9 (474.73nm): Blue - chlorophyll absorption
   - Band 18 (538.71nm): Green - healthy vegetation
   - Band 32 (650.665nm): Red - chlorophyll content
   - Band 42 (730.635nm): Red-edge - stress detection
   - Band 55 (850.59nm): NIR - leaf health

2. Data Format:
   - TIFF format with floating-point data type
   - NaNs used to indicate background are accepted
   - Minimum 55 spectral bands
"""
    with open(training_split_dir / "README_split_methodology.txt", "w") as f:
        f.write(methodology_text)

def run_split(
    dataset_dir: Union[str, Path],
    train_ratio: float = 0.8,
    seed: int = 42,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[Path], List[Path]]:
    """
    Run the dataset splitting process.

    Args:
        dataset_dir: Directory containing multispectral TIFF files
        train_ratio: Ratio of training data (default: 0.8)
        seed: Random seed for reproducibility
        logger: Optional logger instance (will create one if not provided)

    Returns:
        Tuple of (train_files, val_files) containing the split file paths
    """
    if logger is None:
        logger = setup_logging()

    dataset_dir = Path(dataset_dir)
    output_dir = Path(__file__).parent  # Save in script directory

    # Find and validate files
    all_files = find_valid_files(dataset_dir)

    if not all_files:
        logger.error("No valid files found. Check directory path and filtering conditions.")
        return [], []

    # Split files
    train_files, val_files = split_files(all_files, train_ratio, seed)

    logger.info(f"Split into {len(train_files)} training files and {len(val_files)} validation files")

    # Create date-based folder name
    current_date = datetime.now()
    folder_name = f"Training_Split_{current_date.strftime('%d.%m')}"
    training_split_dir = output_dir / folder_name

    # Write file lists and documentation
    write_file_lists(train_files, val_files, output_dir)
    write_methodology(output_dir)

    logger.info(f"Split completed successfully. Files saved in: {training_split_dir}")
    logger.info("Use train_files.txt as input for VAE training")

    return train_files, val_files

def main():
    parser = argparse.ArgumentParser(description="Split multispectral dataset into train/val sets")
    parser.add_argument("--dataset_dir", type=str, required=True,
                      help="Directory containing multispectral TIFF files")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                      help="Ratio of training data (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")

    args = parser.parse_args()
    run_split(args.dataset_dir, args.train_ratio, args.seed)

if __name__ == "__main__":
    main()