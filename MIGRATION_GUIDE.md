# Repository Migration Guide

## Prerequisites

1. Python 3.8+ installed
2. CUDA-compatible GPU with drivers installed
3. Git installed
4. Sufficient disk space for:
   - Repository: ~1GB
   - Dependencies: ~2GB
   - Dataset: Check with data team
   - Model checkpoints: ~10GB

## Setup Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd diffusers
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install rasterio system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev

# macOS
brew install gdal

# Windows
# Download and install GDAL from: https://www.gisinternals.com/release.php
```

5. Verify installation:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import rasterio; print(f'Rasterio version: {rasterio.__version__}')"
```

## Data Preparation

1. Verify data directory structure:
```
data/
├── raw/
│   └── multispectral_tiffs/
│       ├── healthy_leaves/
│       └── stressed_leaves/
└── processed/
    ├── train_files.txt
    ├── val_files.txt
    └── split_stats.txt
```

2. Run dataset split:
```bash
python examples/multispectral/split_dataset.py \
    --dataset_dir /path/to/multispectral/tiffs \
    --train_ratio 0.8 \
    --seed 42
```

3. Verify split outputs:
- Check train_files.txt and val_files.txt exist
- Verify file paths are correct
- Check split_stats.txt for distribution

## Configuration

1. Set up wandb:
```bash
wandb login
```

2. Create config directory:
```bash
mkdir -p configs
```

3. Update paths in training script:
- Check and update model paths
- Verify data paths
- Set logging directory

## Testing

1. Run small test:
```bash
python examples/multispectral/train_multispectral_vae_5ch.py \
    --dataset_dir /path/to/test/data \
    --output_dir ./test_output \
    --num_train_epochs 1 \
    --max_train_steps 10
```

2. Verify outputs:
- Check wandb logging
- Verify model checkpoints
- Check validation images

## Common Issues

1. CUDA out of memory:
- Reduce batch size
- Enable gradient checkpointing
- Use mixed precision training

2. Rasterio errors:
- Verify GDAL installation
- Check file permissions
- Verify TIFF format

3. Path issues:
- Use absolute paths
- Check path separators
- Verify directory permissions

## Support

For issues:
1. Check error logs
2. Verify data format
3. Check system requirements
4. Contact development team

## Next Steps

1. Run full training:
```bash
python examples/multispectral/train_multispectral_vae_5ch.py \
    --dataset_dir /path/to/data \
    --output_dir ./output \
    --num_train_epochs 100
```

2. Monitor training:
- Watch wandb dashboard
- Check GPU utilization
- Monitor disk space

3. Regular backups:
- Save model checkpoints
- Export wandb runs
- Backup configuration 