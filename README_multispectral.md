# Stable Diffusion 3 Multispectral Training

This repository contains the implementation of DreamBooth training for Stable Diffusion 3 with multispectral image support. The implementation is specifically designed to work with 5-channel multispectral data.

updated 9.10.2025

## Overview

This project extends the Hugging Face Diffusers library to support training Stable Diffusion 3 on multispectral imagery. Key features include:

- Adapter-based VAE implementation for 5-channel multispectral data
- Specialized dataloader for multispectral TIFF files
- Integration with Stable Diffusion 3's architecture
- Memory-efficient training pipeline
- Spectral attention mechanism for band importance
- Spectral Angle Mapper (SAM) loss for spectral fidelity

## Repository Structure

```
diffusers/
├── src/
│   └── diffusers/
│       ├── __init__.py
│       └── models/
│           ├── __init__.py
│           └── autoencoders/
│               └── autoencoder_kl_multispectral_adapter.py
├── examples/
│   └── multispectral/
│       ├── split_dataset.py
│       └── train_multispectral_vae_5ch.py
├── multispectral_dataloader.py
├── test_vae_multispectral.py
├── test_multispectral_dataloader.py
├── setup.py
└── requirements.txt
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd diffusers
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e .
```

4. Install additional requirements:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. First, split your dataset using the provided script:
```bash
python examples/multispectral/split_dataset.py \
    --dataset_dir /path/to/multispectral/tiffs \
    --train_ratio 0.8 \
    --seed 42
```

2. Your multispectral data should be organized as follows:
```
/path/to/data/
└── Output Testset Mango/
    └── *.tif  # 5-channel multispectral TIFF files
```

Each TIFF file should contain at least 55 bands of spectral data, with the following bands selected:
- Band 9 (474.73nm): Blue - captures chlorophyll absorption
- Band 18 (538.71nm): Green - reflects well in healthy vegetation
- Band 32 (650.665nm): Red - sensitive to chlorophyll content
- Band 42 (730.635nm): Red-edge - sensitive to stress and early disease
- Band 55 (850.59nm): NIR - strong reflectance in healthy leaves

## Training

1. First, train the multispectral VAE adapter:
```bash
python examples/multispectral/train_multispectral_vae_5ch.py \
    --output_dir /path/to/save/model \
    --num_epochs 100 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --adapter_placement both \
    --use_spectral_attention \
    --use_sam_loss \
    --sam_weight 0.1
```

2. Then, train the DreamBooth model:
```bash
PYTHONPATH=$PYTHONPATH:. accelerate launch examples/dreambooth/train_dreambooth_sd3_multispectral.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers" \
    --instance_data_dir="/path/to/your/data" \
    --output_dir="sd3-dreambooth-multispectral" \
    --instance_prompt="sks leaf with no background" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=2e-6 \
    --lr_scheduler=constant \
    --lr_warmup_steps=0 \
    --max_train_steps=500 \
    --validation_prompt="sks leaf with no background" \
    --validation_epochs=25 \
    --seed=0
```

## Key Components

### 1. Multispectral VAE Adapter
The `AutoencoderKLMultispectralAdapter` class implements a lightweight adapter architecture that:
- Uses pretrained SD3 VAE as backbone
- Adds spectral attention mechanism
- Implements Spectral Angle Mapper (SAM) loss
- Maintains spectral fidelity while leveraging pretrained knowledge

### 2. Multispectral DataLoader
The custom dataloader (`multispectral_dataloader.py`) provides:
- Efficient loading of 5-channel multispectral data
- Per-channel normalization
- Memory-efficient processing
- GPU optimization
- Caching for improved performance

### 3. Training Pipeline
The training pipeline consists of two main components:
1. VAE Adapter Training (`train_multispectral_vae_5ch.py`)
   - Parameter-efficient fine-tuning
   - Spectral attention learning
   - Spectral fidelity preservation
2. DreamBooth Training (`train_dreambooth_sd3_multispectral.py`)
   - Concept learning with multispectral data
   - Integration with pretrained VAE adapter

## Testing

Run the test suite to verify the implementation:
```bash
pytest test_vae_multispectral.py test_multispectral_dataloader.py
```

## Troubleshooting

1. **ImportError for AutoencoderKLMultispectralAdapter**
   - Ensure the package is installed in development mode
   - Verify the class is properly imported in `__init__.py` files

2. **ModuleNotFoundError for multispectral_dataloader**
   - Set PYTHONPATH to include the current directory
   - Verify the dataloader file is in the correct location

3. **Data Loading Issues**
   - Check TIFF file format and band count
   - Verify file permissions and paths
   - Ensure sufficient disk space and memory

4. **Training Issues**
   - Check GPU memory usage
   - Verify batch size settings
   - Monitor spectral attention weights
   - Check loss term balancing
