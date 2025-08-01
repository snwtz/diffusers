MULTISPECTRAL VAE TRAINING PIPELINE - DIGITAL ARCHIVE
====================================================

This digital archive contains the complete implementation of a multispectral VAE (MSAE) training pipeline
for adapting Stable Diffusion 3 (SD3) to generate synthetic 5-channel multispectral plant imagery.
The pipeline enables parameter-efficient fine-tuning of pretrained SD3 VAE for hyperspectral data
while maintaining spectral fidelity and SD3 compatibility.

RESEARCH CONTEXT
---------------
This work addresses the challenge of adapting pretrained diffusion models for multispectral
plant imaging applications. The core innovation is a lightweight adapter architecture that
bridges 5-channel hyperspectral data to the 3-channel RGB format expected by SD3, enabling
synthetic multispectral image generation through DreamBooth fine-tuning.

ARCHIVE STRUCTURE
================

1. CORE IMPLEMENTATION FILES
---------------------------

examples/multispectral/
├── vae_multispectral_dataloader.py
│   └── Multispectral image dataloader for VAE training
│       - Handles 5-channel hyperspectral TIFF files
│       - Extracts specific bands (9, 18, 32, 42, 55) for plant analysis
│       - Implements background masking using NaN values
│       - Per-channel normalization to [-1, 1] range
│       - Supports train/val splits via file lists
│
├── train_multispectral_vae_5ch.py
│   └── Main training script for multispectral VAE adapter
│       - Multi-objective loss (MSE + SAM + spectral signature)
│       - Parameter-efficient fine-tuning (frozen RGB backbone)
│       - Monitoring and logging with Weights & Biases integration
│       - Early stopping and checkpointing
│       - SSIM computation on masked regions
│
└── autoencoder_kl_multispectral_adapter.py
    └── Multispectral VAE adapter architecture
        - Lightweight adapters bridging 5→3→5 channels
        - Spectral attention mechanism for learned band importance
        - Frozen RGB backbone for parameter efficiency
        - Masked loss computation for leaf-focused training
        - SD3 pipeline compatibility


4. CONFIGURATION AND SETUP
-------------------------

The pipeline requires:
- Python 3.8+ with PyTorch, diffusers, rasterio
- CUDA-compatible GPU for training
- Hyperspectral TIFF files with at least 55 bands
- Weights & Biases account for experiment tracking (optional)

5. LOGGING AND MONITORING
-------------------------

- Training/validation loss (total, per-channel MSE, global SAM)
- Learning rate and gradient norm
- Per-band MSE and SSIM (5 spectral bands)
- Spectral signature comparison with reference
- Scale convergence monitoring
- Output range statistics
- Model health (memory usage, NaN detection)
- Band importance weights (spectral attention)

6. USAGE WORKFLOW
-----------------

1. Data Preparation:
   - Organize hyperspectral TIFF files
   - Create train/val split files using split_dataset.py
   - Ensure files contain required bands (9, 18, 32, 42, 55)

2. VAE Training:
   - Run train_multispectral_vae_5ch.py with appropriate arguments
   - Monitor training via Weights & Biases dashboard
   - Check output range statistics and scale convergence
   - Save best model checkpoint

3. DreamBooth Integration:
   - Use trained VAE in SD3 + DreamBooth pipeline
   - Generate synthetic multispectral images
   - Evaluate spectral fidelity and plant health analysis



10. FILE DESCRIPTIONS
--------------------

vae_multispectral_dataloader.py:
- Dataset class for loading 5-channel multispectral TIFF files
- Background masking using NaN values
- Per-channel normalization and padding strategies
- Factory function for creating train/val dataloaders

train_multispectral_vae_5ch.py:
- Complete training pipeline with multi-objective loss
- Advanced monitoring (SSIM, scale convergence, band importance)
- Weights & Biases integration for experiment tracking
- Early stopping, checkpointing, and model saving

autoencoder_kl_multispectral_adapter.py:
- Core adapter architecture extending AutoencoderKL
- Spectral attention mechanism for band weighting
- Masked loss computation for leaf-focused training
- SD3 pipeline compatibility with DecoderOutput format

11. CITATION AND ACKNOWLEDGMENTS
--------------------------------

This work builds upon:
- Stable Diffusion 3 (Stability AI)
- Diffusers library (Hugging Face)
- DreamBooth fine-tuning methodology
- Spectral Angle Mapper for multispectral analysis
