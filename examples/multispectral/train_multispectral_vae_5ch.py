"""
Training script for Multispectral VAE Adapter

This script implements the training pipeline for the thesis's core methodological contribution:
a lightweight adapter-based multispectral VAE architecture. It serves as the pretraining step
for integrating a custom VAE into the Stable Diffusion 3 + DreamBooth pipeline for generating
synthetic multispectral plant imagery.

IMPORTANT: This script is configured to use 16 latent channels (matching AutoencoderKL/HF/SD3(??) default),
which provides more capacity to encode multispectral information compared to the standard 4 channels.
This is beneficial for multispectral data as it allows the model to preserve more spectral detail
while maintaining compatibility with the SD3 transformer architecture.

Data Flow Summary:
------------------
- Input: Preprocessed 5-channel plant images and masks (from vae_multispectral_dataloader.py)
- Model: Adapters map 5-channel input to 3-channel HF's AutoencoderKL backbone, then back to 5-channel output
- Output: Reconstructed 5-channel image, with losses computed for both spatial and spectral fidelity
- Loss: Multi-objective (MSE + SAM), with mask-aware background handling
- Logging: Per-epoch metrics, band importance, and SSIM for scientific analysis

Training Strategy:
The model should naturally learn to produce outputs in the [-1, 1] range because:
Input data is normalized to [-1, 1]
Loss function penalizes reconstruction error
The model will learn to match the input distribution

However:
- The nonlinear processes (spectral attention, SiLU) contain important spectral information
- spectral relationships learned by the attention mechanism need to be preserved (no hard clamping or tanh activation to force data range [-1/1])
- 

Thesis Context and Training Workflow:
----------------------------------
1. Research Pipeline:
   - Pretraining: This script trains the multispectral VAE adapter
   - Fine-tuning: DreamBooth adapts SD3 for multispectral generation
   - Evaluation: Spectral fidelity

2. Data Processing:
   The training pipeline handles 5 biologically relevant spectral bands:
   - Band 9 (474.73nm): Blue - captures chlorophyll absorption
   - Band 18 (538.71nm): Green - reflects well in healthy vegetation
   - Band 32 (650.665nm): Red - sensitive to chlorophyll content
   - Band 42 (730.635nm): Red-edge - sensitive to stress and early disease
   - Band 55 (850.59nm): NIR - strong reflectance in healthy leaves

3. Background Handling:
   - NaN values in TIFF files represent background (cut-out regions)
   - Binary masks (1 for leaf, 0 for background) are generated
   - Loss computation primarily considers leaf regions
   - Background contributes softly (10%) to the total loss
   - Model focuses solely on leaf features
   - No background inpainting or interpolation

4. Training Features:
   a) Data Management:
      - Custom 5-channel MultispectralDataset
      - Deterministic train/val splitting
      - Spectral normalization pipeline
      - Memory-efficient loading
      - Background masking

   b) Loss Computation:
      - Per-band MSE for spatial fidelity
      - SAM loss for spectral signature preservation
      - Masked loss computation: full weight on leaf, soft penalty on background (10%)
      - Configurable loss weighting
      - Band-specific loss tracking

   c) Training Optimization:
      - Parameter isolation via get_trainable_params()
      - Learning rate scheduling with warmup
      - Early stopping on validation plateau
      - EMA model averaging
      - Gradient clipping

4. Validation and Monitoring:
   a) Per-epoch Validation:
      - Per-band MSE tracking
      - SAM loss computation
      - Spectral attention weights
      - Reconstruction quality

   b) Scientific Logging:
      - Weights & Biases integration
      - Automatic experiment logging with wandb (setup inside `train()`)
      - Band importance visualization
      - Spectral signature plots
      - Training dynamics analysis

Development and Testing Strategy:
------------------------------
1. Dataset Preparation:
   - split_dataset.py creates deterministic splits
   - Fixed random seed for reproducibility
   - Train/val files for traceability
   - Healthy leaf sample filtering

2. Initial Testing Protocol:
   a) Small-Scale Overfit Test:
      - Train on 1-2 samples
      - Disable dropout
      - Monitor per-band MSE
      - Visualize reconstructions
      Expected: Near-perfect reconstructions in <100 iterations

   b) Validation Metrics:
      - Per-band MSE tracking
      - SAM loss components
      - Spectral attention weights
      - Reconstruction quality

3. Training Configuration:
   - Batch size optimization
   - Learning rate selection
   - Loss weight tuning
   - Early stopping patience

Scientific Contributions:
----------------------
1. Training Methodology:
   - Parameter-efficient fine-tuning
   - Spectral-aware optimization
   - Band importance analysis
   - Spectral fidelity preservation

2. Validation Framework:
   - Spectral quality metrics
   - Band correlation analysis
   - Reconstruction evaluation
   - Scientific visualization

3. Integration Strategy:
   - SD3 compatibility
   - DreamBooth workflow
   - Spectral concept learning
   - Plant health analysis

Usage:
    # First, split the dataset:
    python split_dataset.py \
        --dataset_dir /path/to/multispectral/tiffs \


    # Then, train the VAE:
    python examples\multispectral\train_multispectral_vae_5ch.py --train_file_list "C:/Users/NOcsPS-440g/Desktop/Zina/diffusers/examples/multispectral/Training_Split_18.06/train_files.txt" --val_file_list "C:/Users/NOcsPS-440g/Desktop/Zina/diffusers/examples/multispectral/Training_Split_18.06/val_files.txt" --output_dir "C:/Users/NOcsPS-440g/Desktop/Zina/diffusers/examples/multispectral/Training_Split_18.06" --base_model_path "C:/Users/NOcsPS-440g/Desktop/Zina/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py" --num_epochs 100 --batch_size 8 --learning_rate 1e-4 --adapter_placement both --use_spectral_attention --use_sam_loss --sam_weight 0.1 --warmup_ratio 0.1 --early_stopping_patience 10 --max_grad_norm 1.0 --use_saturation_penalty
NOTE: add to your CLI call: --use_saturation_penalty

    # Testing
    python examples\multispectral\train_multispectral_vae_5ch.py --train_file_list "C:/Users/NOcsPS-440g/Desktop/Zina/diffusers/examples/multispectral/Training_Split_18.06/train_files.txt" --val_file_list "C:/Users/NOcsPS-440g/Desktop/Zina/diffusers/examples/multispectral/Training_Split_18.06/val_files.txt" --output_dir "C:/Users/NOcsPS-440g/Desktop/Zina/diffusers/examples/multispectral/Training_Split_18.06" --base_model_path "C:/Users/NOcsPS-440g/Desktop/Zina/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py" --num_epochs 2 --batch_size 1 --learning_rate 1e-4 --adapter_placement both --use_spectral_attention --use_sam_loss --sam_weight 0.1 --warmup_ratio 0.1 --early_stopping_patience 1 --max_grad_norm 1.0 --num_workers 0 --use_saturation_penalty

VAE Loading Note:
   RGB VAE weights from SD3 could not be loaded directly via `from_pretrained()` using a config object,
    because the class `AutoencoderKLMultispectralAdapter` expects unpacked keyword arguments, not a config instance.
    Successfully loaded the RGB VAE weights from SD3 by explicitly:
        1. Using `AutoencoderKL.from_config()` to parse the original RGB config.json.
        2. Loading pretrained SD3 VAE weights with `load_state_dict`.
        3. Passing the loaded AutoencoderKL instance as a `base_model` argument to our `AutoencoderKLMultispectralAdapter`.

This approach bypasses issues with conflicting `from_pretrained` calls and allows reuse of the SD3 VAE backbone with frozen weights.
"""

import os
import torchvision.utils as vutils
from PIL import Image
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from datetime import datetime
import wandb
import shutil
from typing import Tuple
import numpy as np  # Ensure numpy is imported
from skimage.metrics import structural_similarity as ssim
import time # Added for timing

from diffusers import AutoencoderKLMultispectralAdapter
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from vae_multispectral_dataloader import create_vae_dataloaders
from training_logger import create_training_logger

def setup_logging(args):
    # Create logs directory for experiment traceability and reproducibility
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Setup file handler
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Setup logger
    logger = logging.getLogger('multispectral_vae')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def log_training_metrics(logger, epoch, train_losses, val_losses, band_importance, args, output_range_stats=None):
    # Logs all relevant metrics for scientific reporting and experiment tracking
    logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
    logger.info("Training Metrics:")
    logger.info(f"  Total Loss: {train_losses.get('total_loss', float('nan')):.4f}")
    logger.info("  Per-channel MSE:")
    for i, loss in enumerate(train_losses.get('mse_per_channel', [])):
        logger.info(f"    Band {i+1}: {loss:.4f}")
    if 'sam' in train_losses:
        logger.info(f"  SAM Loss: {train_losses['sam']:.4f}")

    logger.info("Validation Metrics:")
    logger.info(f"  Total Loss: {val_losses.get('total_loss', float('nan')):.4f}")
    logger.info("  Per-channel MSE:")
    for i, loss in enumerate(val_losses.get('mse_per_channel', [])):
        logger.info(f"    Band {i+1}: {loss:.4f}")
    if 'sam' in val_losses:
        logger.info(f"  SAM Loss: {val_losses['sam']:.4f}")

    logger.info("Band Importance:")
    for band, importance in band_importance.items():
        logger.info(f"  {band}: {importance:.4f}")
    
    # Log mask statistics if available
    if 'mask_stats' in train_losses:
        train_mask_stats = train_losses['mask_stats']
        logger.info("Training Mask Statistics:")
        logger.info(f"  Coverage: {train_mask_stats['coverage']:.4f}")
        logger.info(f"  Valid pixels: {train_mask_stats['valid_pixels']}/{train_mask_stats['total_pixels']}")
    
    if 'mask_stats' in val_losses:
        val_mask_stats = val_losses['mask_stats']
        logger.info("Validation Mask Statistics:")
        logger.info(f"  Coverage: {val_mask_stats['coverage']:.4f}")
        logger.info(f"  Valid pixels: {val_mask_stats['valid_pixels']}/{val_mask_stats['total_pixels']}")
    
    # Log scale monitoring information if available
    if 'scale_monitoring' in train_losses:
        scale_info = train_losses['scale_monitoring']
        logger.info("Scale Monitoring (Training):")
        logger.info(f"  Global Scale: {scale_info['scale_value']:.6f}")
        logger.info(f"  Scale Mean: {scale_info['scale_mean']:.6f}")
        logger.info(f"  Scale Std: {scale_info['scale_std']:.6f}")
        logger.info(f"  Is Converged: {scale_info['is_converged']}")
        logger.info(f"  Convergence Rate: {scale_info['convergence_rate']:.6f}")
        logger.info(f"  History Length: {scale_info['history_length']}")
        if scale_info['warnings']:
            logger.warning("  Scale Warnings:")
            for warning in scale_info['warnings']:
                logger.warning(f"    {warning}")
        if scale_info['recommendations']:
            logger.info("  Scale Recommendations:")
            for rec in scale_info['recommendations']:
                logger.info(f"    {rec}")
    
    if 'scale_monitoring' in val_losses:
        scale_info = val_losses['scale_monitoring']
        logger.info("Scale Monitoring (Validation):")
        logger.info(f"  Global Scale: {scale_info['scale_value']:.6f}")
        logger.info(f"  Scale Mean: {scale_info['scale_mean']:.6f}")
        logger.info(f"  Scale Std: {scale_info['scale_std']:.6f}")
        logger.info(f"  Is Converged: {scale_info['is_converged']}")
        logger.info(f"  Convergence Rate: {scale_info['convergence_rate']:.6f}")
        logger.info(f"  History Length: {scale_info['history_length']}")
    
    # Log output range statistics if provided
    if output_range_stats:
        logger.info("Output Range Statistics:")
        logger.info(f"  Global min: {output_range_stats['global_min']:.4f}")
        logger.info(f"  Global max: {output_range_stats['global_max']:.4f}")
        logger.info(f"  Global mean: {output_range_stats['global_mean']:.4f}")
        logger.info(f"  Global std: {output_range_stats['global_std']:.4f}")
        logger.info("  Per-band ranges:")
        for i, (min_val, max_val, mean_val, std_val) in enumerate(output_range_stats['per_band']):
            logger.info(f"    Band {i+1}: [{min_val:.4f}, {max_val:.4f}] mean={mean_val:.4f} std={std_val:.4f}")
        
        # Warn if output range is significantly different from expected [-1, 1]
        if output_range_stats['global_min'] < -2.0 or output_range_stats['global_max'] > 2.0:
            logger.warning(f"Output range [{output_range_stats['global_min']:.4f}, {output_range_stats['global_max']:.4f}] "
                          f"is outside expected [-1, 1] range. Consider adjusting SSIM data_range parameter.")

def setup_wandb(args):
    """Initializes Weights & Biases for experiment tracking and reproducibility."""
    try:
        # Create a descriptive run name
        run_name = f"multispectral_vae_{args.adapter_placement}"
        if args.use_spectral_attention:
            run_name += "_spectral_attn"
        if args.use_sam_loss:
            run_name += "_sam"
        run_name += f"_lr{args.learning_rate}_bs{args.batch_size}"
        
        wandb.init(
            project="multispectral-vae",
            name=run_name,
            config={
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "adapter_placement": args.adapter_placement,
                "use_spectral_attention": args.use_spectral_attention,
                "use_sam_loss": args.use_sam_loss,
                "sam_weight": args.sam_weight,
                "warmup_ratio": args.warmup_ratio,
                "early_stopping_patience": args.early_stopping_patience,
                "max_grad_norm": args.max_grad_norm,
                "base_model_path": args.base_model_path,
                "num_workers": args.num_workers,
                "save_every": args.save_every,
                "model_type": "multispectral_vae_adapter",
                "input_channels": 5,
                "output_channels": 5,
                "latent_channels": 16,  # SD3 default
            },
            tags=["multispectral", "vae", "plant-health", "spectral-imaging"]
        )
        return True
    except Exception as e:
        # Use logger if available, otherwise print
        try:
            logger = logging.getLogger('multispectral_vae')
            logger.warning(f"Failed to initialize wandb: {e}")
        except:
            print(f"Warning: Failed to initialize wandb: {e}")
        return False

def log_to_wandb(epoch, train_losses, val_losses, band_importance, batch, reconstruction, model, output_range_stats=None, ssim_per_band=None, current_lr=None, grad_norm=None):
    """Logs metrics and sample images to Weights & Biases for visualization and analysis."""
    # Check if wandb is initialized
    if not wandb.run:
        return
    
    try:
        # Prepare metrics data
        wandb_log_data = {
            "epoch": epoch,
            "train_total_loss": train_losses.get('total_loss', float('nan')),
            "val_total_loss": val_losses.get('total_loss', float('nan')),
            **{f"train_mse_band_{i}": loss for i, loss in enumerate(train_losses.get('mse_per_channel', []))},
            **{f"val_mse_band_{i}": loss for i, loss in enumerate(val_losses.get('mse_per_channel', []))},
            **{f"band_importance_{k}": v for k, v in band_importance.items()},
        }
        
        # Add SSIM metrics if available
        if ssim_per_band is not None:
            wandb_log_data.update({
                "avg_ssim": np.mean(ssim_per_band),
                **{f"ssim_band_{i}": ssim_val for i, ssim_val in enumerate(ssim_per_band)}
            })
        
        # Add learning rate if available
        if current_lr is not None:
            wandb_log_data["learning_rate"] = current_lr
        
        # Add gradient norm if available
        if grad_norm is not None:
            wandb_log_data["gradient_norm"] = grad_norm
        
        # Add SAM loss if available
        if 'sam' in train_losses and 'sam' in val_losses:
            wandb_log_data.update({
                "train_sam_loss": train_losses['sam'],
                "val_sam_loss": val_losses['sam']
            })
        
        # Add mask statistics if available
        if 'mask_stats' in train_losses:
            train_mask_stats = train_losses['mask_stats']
            wandb_log_data.update({
                "mask/train_coverage": train_mask_stats['coverage'],
                "mask/train_valid_pixels": train_mask_stats['valid_pixels'],
                "mask/train_total_pixels": train_mask_stats['total_pixels']
            })
        
        if 'mask_stats' in val_losses:
            val_mask_stats = val_losses['mask_stats']
            wandb_log_data.update({
                "mask/val_coverage": val_mask_stats['coverage'],
                "mask/val_valid_pixels": val_mask_stats['valid_pixels'],
                "mask/val_total_pixels": val_mask_stats['total_pixels']
            })
        
        # Add scale monitoring metrics if available
        if 'scale_monitoring' in train_losses:
            scale_info = train_losses['scale_monitoring']
            wandb_log_data.update({
                "scale/train_global_scale": scale_info['scale_value'],
                "scale/train_scale_mean": scale_info['scale_mean'],
                "scale/train_scale_std": scale_info['scale_std'],
                "scale/train_is_converged": scale_info['is_converged'],
                "scale/train_convergence_rate": scale_info['convergence_rate'],
                "scale/train_history_length": scale_info['history_length'],
                "scale/train_is_collapsed": scale_info['is_collapsed'],
                "scale/train_is_exploded": scale_info['is_exploded']
            })
        
        if 'scale_monitoring' in val_losses:
            scale_info = val_losses['scale_monitoring']
            wandb_log_data.update({
                "scale/val_global_scale": scale_info['scale_value'],
                "scale/val_scale_mean": scale_info['scale_mean'],
                "scale/val_scale_std": scale_info['scale_std'],
                "scale/val_is_converged": scale_info['is_converged'],
                "scale/val_convergence_rate": scale_info['convergence_rate'],
                "scale/val_history_length": scale_info['history_length'],
                "scale/val_is_collapsed": scale_info['is_collapsed'],
                "scale/val_is_exploded": scale_info['is_exploded']
            })
        
        # Add output range statistics if available
        if output_range_stats:
            wandb_log_data.update({
                "output_range/global_min": output_range_stats['global_min'],
                "output_range/global_max": output_range_stats['global_max'],
                "output_range/global_mean": output_range_stats['global_mean'],
                "output_range/global_std": output_range_stats['global_std'],
                "output_range/warning": output_range_stats['range_warning']
            })
            
            # Add per-band range statistics
            for i, (min_val, max_val, mean_val, std_val) in enumerate(output_range_stats['per_band']):
                wandb_log_data.update({
                    f"output_range/band_{i}_min": min_val,
                    f"output_range/band_{i}_max": max_val,
                    f"output_range/band_{i}_mean": mean_val,
                    f"output_range/band_{i}_std": std_val,
                })

        # Log learnable output scaling parameters (if available)
        if hasattr(model.output_adapter, 'output_scale'):
            wandb_log_data["output_adapter/output_scale"] = model.output_adapter.output_scale.item()
        if hasattr(model.output_adapter, 'output_bias'):
            wandb_log_data["output_adapter/output_bias"] = model.output_adapter.output_bias.item()
        
        # Log metrics
        wandb.log(wandb_log_data)
        
        # Log sample images (first sample, first band as grayscale)
        if batch is not None and reconstruction is not None:
            try:
                # Convert tensors to PIL images for wandb logging
                # Normalize from [-1, 1] to [0, 255] range
                orig_band0 = (batch[0, 0].detach().cpu().numpy() + 1.0) * 127.5
                recon_band0 = (reconstruction[0, 0].detach().cpu().numpy() + 1.0) * 127.5
                
                # Ensure values are in valid range
                orig_band0 = np.clip(orig_band0, 0, 255).astype(np.uint8)
                recon_band0 = np.clip(recon_band0, 0, 255).astype(np.uint8)
                
                # Convert to PIL images
                orig_pil = Image.fromarray(orig_band0, mode='L')  # Grayscale
                recon_pil = Image.fromarray(recon_band0, mode='L')  # Grayscale
                
                # Log images to wandb
                wandb.log({
                    "images/original_band0": wandb.Image(orig_pil, caption=f"Original Band 0 - Epoch {epoch+1}"),
                    "images/reconstructed_band0": wandb.Image(recon_pil, caption=f"Reconstructed Band 0 - Epoch {epoch+1}")
                })
            except Exception as img_error:
                # Use logger if available, otherwise print
                try:
                    logger = logging.getLogger('multispectral_vae')
                    logger.warning(f"Failed to log images to wandb: {img_error}")
                except:
                    print(f"Warning: Failed to log images to wandb: {img_error}")
            
    except Exception as e:
        # Log error but don't crash training
        try:
            logger = logging.getLogger('multispectral_vae')
            logger.warning(f"Failed to log to wandb: {e}")
        except:
            print(f"Warning: Failed to log to wandb: {e}")
        # You could also use logger.warning here if logger is available

def save_checkpoint(model, optimizer, scheduler, epoch, loss, is_best, args, best_val_loss=None, patience_counter=None):
    """Save model checkpoint.
 
    - Checkpointing ensures that the best model is preserved for downstream analysis and reproducibility.
    - Regular checkpoints allow for recovery from interruptions and support ablation studies.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter
    }

    # Save regular checkpoint
    checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{epoch+1}")
    model.save_pretrained(checkpoint_path)
    torch.save(checkpoint, os.path.join(checkpoint_path, 'training_state.pt'))

    # Save best model if needed
    if is_best:
        best_model_path = os.path.join(args.output_dir, 'best_model')
        if os.path.exists(best_model_path):
            shutil.rmtree(best_model_path)
        model.save_pretrained(best_model_path)
        torch.save(checkpoint, os.path.join(best_model_path, 'training_state.pt'))

def count_parameters(model):
    """Count trainable and non-trainable parameters in the model to report model complexity 
    and for comparing different architectures.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params

    return {
        'trainable': trainable_params,
        'non_trainable': non_trainable_params,
        'total': total_params,
        'trainable_percentage': (trainable_params / total_params) * 100
    }

def log_parameter_counts(model, logger, wandb_log=True):
    """Log model parameter counts to logger and wandb."""
    param_counts = count_parameters(model)

    logger.info("Model Parameter Counts:")
    logger.info(f"  Trainable parameters: {param_counts['trainable']:,}")
    logger.info(f"  Non-trainable parameters: {param_counts['non_trainable']:,}")
    logger.info(f"  Total parameters: {param_counts['total']:,}")
    logger.info(f"  Trainable percentage: {param_counts['trainable_percentage']:.2f}%")

    if wandb_log and wandb.run:
        try:
            wandb.log({
                "model/trainable_params": param_counts['trainable'],
                "model/non_trainable_params": param_counts['non_trainable'],
                "model/total_params": param_counts['total'],
                "model/trainable_percentage": param_counts['trainable_percentage']
            })
        except Exception as e:
            logger.warning(f"Failed to log parameter counts to wandb: {e}")

def compute_output_range_stats(reconstruction_batch: torch.Tensor) -> dict:
    """
    Compute comprehensive statistics about the output range of the VAE decoder.
    
    This function helps monitor the actual output range during training to:
    1. Detect if outputs are within expected [-1, 1] range
    2. Identify per-band range variations
    3. Provide guidance for SSIM data_range parameter adjustment
    4. Monitor for potential numerical instability
    
    POTENTIAL RUNTIME ISSUES AND SOLUTIONS:
    --------------------------------------
    1. Output Range Mismatch:
       - Issue: Adapter nonlinearities produce outputs outside [-1, 1] range
       - Impact: SSIM computation assumes [-1, 1] range (data_range=2.0)
       - Solution: Monitor actual range and adjust SSIM data_range parameter
       - Detection: This function provides range statistics and warnings
    
    2. Numerical Instability:
       - Issue: Very large or small values from nonlinear transformations
       - Impact: Loss computation, gradient explosion, training instability
       - Solution: Monitor global min/max and add gradient clipping
       - Detection: range_warning flag when values outside [-2, 2]
    
    3. Per-band Range Variations:
       - Issue: Different spectral bands may have different output ranges
       - Impact: Inconsistent loss scaling across bands
       - Solution: Monitor per-band statistics and consider band-specific normalization
       - Detection: Per-band min/max/mean/std statistics
    
    4. SSIM Parameter Adjustment:
       - Issue: SSIM data_range parameter mismatch with actual output range
       - Impact: Incorrect SSIM scores, misleading validation metrics
       - Solution: Use actual output range to compute data_range parameter
       - Detection: Compare actual range with assumed [-1, 1] range
    
    Args:
        reconstruction_batch: Tensor of shape (B, C, H, W) from VAE decoder
        
    Returns:
        Dictionary containing range statistics:
        - global_min/max/mean/std: Overall statistics across all bands
        - per_band: List of (min, max, mean, std) for each band
        - range_warning: Boolean indicating if range is outside [-2, 2]
    """
    with torch.no_grad():
        # Global statistics across all bands
        global_min = reconstruction_batch.min().item()
        global_max = reconstruction_batch.max().item()
        global_mean = reconstruction_batch.mean().item()
        global_std = reconstruction_batch.std().item()
        
        # Per-band statistics
        per_band_stats = []
        for band_idx in range(reconstruction_batch.shape[1]):
            band_data = reconstruction_batch[:, band_idx]
            band_min = band_data.min().item()
            band_max = band_data.max().item()
            band_mean = band_data.mean().item()
            band_std = band_data.std().item()
            per_band_stats.append((band_min, band_max, band_mean, band_std))
        
        # Check for range warnings
        range_warning = global_min < -2.0 or global_max > 2.0
        
        return {
            'global_min': global_min,
            'global_max': global_max,
            'global_mean': global_mean,
            'global_std': global_std,
            'per_band': per_band_stats,
            'range_warning': range_warning
        }

def prepare_dataset(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare the dataset using the split files created by split_dataset.py.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (train_loader, val_loader)

    Implementation Notes:
    -------------------
    1. File List Management:
       - Uses split files from split_dataset.py for deterministic, reproducible splits.
       - Ensures data consistency and traceability.

    2. Dataloader Configuration:
       - Uses VAE-specific dataloader module.
       - Optimizes for GPU training and efficient data loading.

    3. Data Quality Assurance:
       - Inspects the first batch for NaNs/Infs to ensure data quality before training.
       - Logs anomalies for debugging and reproducibility.

    4. Error Handling:
       - Validates split files exist and provides clear error messages.
       - Ensures training stability and scientific rigor.
    """
    # Get split files from args
    train_list = Path(args.train_file_list)
    val_list = Path(args.val_file_list)

    # Validate split files exist
    if not train_list.exists():
        raise FileNotFoundError(f"Training file list not found: {train_list}")
    if not val_list.exists():
        raise FileNotFoundError(f"Validation file list not found: {val_list}")

    # Create dataloaders using the new VAE dataloader module
    # This factory function ensures consistent configuration and optimal settings
    try:
        train_loader, val_loader = create_vae_dataloaders(
            train_list_path=str(train_list),
            val_list_path=str(val_list),
            batch_size=args.batch_size,
            resolution=512,  # Fixed resolution for VAE training
            num_workers=args.num_workers,
            use_cache=True,  # Enable caching for repeated access
            prefetch_factor=2 if args.num_workers > 0 else None,  # Optimize data loading
            persistent_workers=args.num_workers > 0  # Efficient worker management
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create dataloaders: {e}")

    """
    # Sample inspection — checking first training batch for NaNs
    logger = logging.getLogger('multispectral_vae')
    logger.info(f"Sample inspection — checking first training batch for NaNs")
    # New debug check: inspect for NaNs before any sanitization or clamping
    for sample_batch, mask in train_loader:
        logger.info("Checking for NaNs/Infs in first batch (pre-sanitization)...")
        raw_nan_check = torch.isnan(sample_batch)
        if raw_nan_check.any():
            logger.warning(f"NaNs detected in raw first batch before sanitization. NaN count: {raw_nan_check.sum().item()}")
            for channel_idx in range(sample_batch.shape[1]):
                channel_nans = raw_nan_check[:, channel_idx].sum()
                if channel_nans > 0:
                    logger.warning(f"  NaNs in channel {channel_idx}: {channel_nans.item()} total")
        else:
            logger.info("First training batch is free of NaNs (pre-sanitization).")
        break
    """
    return train_loader, val_loader

    
def load_checkpoint(checkpoint_path: str, model, optimizer, scheduler, device):
    """
    Load training state from a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load tensors on
    
    Returns:
        Tuple of (start_epoch, best_val_loss, patience_counter)
    """
    import logging
    logger = logging.getLogger(__name__)
    checkpoint_file = os.path.join(checkpoint_path, 'training_state.pt')
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
    
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model weights from epoch {checkpoint['epoch']}")
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    logger.info("Loaded optimizer state")
    
    # Load scheduler state
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    logger.info("Loaded scheduler state")
    
    # Extract training state
    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    patience_counter = checkpoint.get('patience_counter', 0)
    
    logger.info(f"Resuming from epoch {start_epoch} with best validation loss: {best_val_loss:.6f}")
    
    return start_epoch, best_val_loss, patience_counter

def log_training_progress_to_wandb(step, total_loss, current_lr, grad_norm=None, log_interval=10):
    """Log training progress to wandb during training loop."""
    if wandb.run and step % log_interval == 0:
        try:
            log_data = {
                "train_step": step,
                "train_loss": total_loss.item() if hasattr(total_loss, 'item') else total_loss,
                "learning_rate": current_lr
            }
            if grad_norm is not None:
                log_data["gradient_norm"] = grad_norm
            wandb.log(log_data)
        except Exception as e:
            # Silently fail to avoid disrupting training
            pass

def train(args: argparse.Namespace) -> None:
    """
    Main training function.

    Implementation Notes:
    -------------------
    1. Model Initialization:
       - Loads pretrained SD3 model and configures multispectral adapters.
       - Freezes backbone for parameter-efficient fine-tuning (only adapters are trainable).
       - Logs parameter counts for scientific reporting.

    2. Training Pipeline:
       - Uses cosine learning rate schedule with warmup for stable convergence.
       - Implements EMA model averaging for improved generalization.
       - Applies gradient clipping to prevent exploding gradients.
       - Supports early stopping and checkpointing for robust model selection.
       - Handles background masking throughout the pipeline for leaf-focused learning.

    3. Loss Computation:
       - Per-band MSE for spatial fidelity.
       - Optional SAM loss for spectral signature preservation.
       - Masked loss computation: only leaf regions (mask==1) contribute fully; background (mask==0) is softly penalized.
       - Combination of MSE and SAM loss is critical for balancing spatial and spectral fidelity in scientific applications.

    4. Validation Strategy:
       - Per-epoch validation with per-band MSE, SAM loss, and SSIM metrics.
       - SSIM is computed per-band and only on masked (leaf) regions to ensure validity.
       - Tracks best model and implements early stopping to prevent overfitting.
       - Saves checkpoints for reproducibility and analysis.

    5. Monitoring and Logging:
       - Comprehensive logging of all metrics, band importance, and training dynamics.
       - Weights & Biases integration for experiment tracking and visualization.
       - Logs all hyperparameters and results for scientific reproducibility.
    """

    # Setup logging first (after output_dir is set)
    logger = setup_logging(args)
    logger.info(f"Starting multispectral VAE training with output directory: {args.output_dir}")
    
    # Setup training logger for compressed metrics
    training_logger = create_training_logger(args.output_dir, "multispectral_vae_adapter")
    
    # Log training configuration
    config = {
        "base_model_path": args.base_model_path,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "adapter_placement": args.adapter_placement,
        "use_spectral_attention": args.use_spectral_attention,
        "use_sam_loss": args.use_sam_loss,
        "sam_weight": args.sam_weight,
        "use_saturation_penalty": args.use_saturation_penalty,
        "warmup_ratio": args.warmup_ratio,
        "early_stopping_patience": args.early_stopping_patience,
        "max_grad_norm": args.max_grad_norm,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    training_logger.log_config(config)
    
    # Setup persistent directory for saving image samples
    image_output_dir = os.path.join(args.output_dir, "image_samples")
    os.makedirs(image_output_dir, exist_ok=True)

    # Setup wandb early to avoid UnboundLocalError
    wandb_initialized = False
    if not args.disable_wandb:
        wandb_initialized = setup_wandb(args)
        if wandb_initialized:
            logger.info("Wandb initialized successfully")
        else:
            logger.warning("Wandb initialization failed - continuing without wandb logging")
    else:
        logger.info("Wandb logging disabled by user")

    # Load base SD3 VAE weights and inject fresh multispectral adapters
    try:
        logger.info(f"Loading base model from: {args.base_model_path}")
        model = AutoencoderKLMultispectralAdapter.from_pretrained(
            pretrained_model_name_or_path=args.base_model_path,
            subfolder=args.subfolder,
            adapter_placement=args.adapter_placement,
            use_spectral_attention=args.use_spectral_attention,
            use_sam_loss=args.use_sam_loss,
            in_channels=5,  # Force 5 channels for multispectral input
            out_channels=5,   # Force 5 channels for multispectral output
            use_saturation_penalty=args.use_saturation_penalty,
        )
        logger.info("Successfully loaded base model")
        
        # Log the configuration to verify it's set correctly
        logger.info(f"VAE Configuration:")
        logger.info(f"  - in_channels: {model.config.in_channels}")
        logger.info(f"  - out_channels: {model.config.out_channels}")
        logger.info(f"  - latent_channels: {model.config.latent_channels}")
        logger.info(f"  - adapter_placement: {model.adapter_placement}")
        logger.info(f"  - use_spectral_attention: {model.use_spectral_attention}")
        # Explicit assertion and log for input/output adapter channels (safer)
        assert model.input_adapter.in_channels == 5, f"Input adapter expects {model.input_adapter.in_channels} channels, expected 5!"
        assert model.output_adapter.out_channels == 5, f"Output adapter produces {model.output_adapter.out_channels} channels, expected 5!"
        logger.info(f"[CHECK] Adapter input/output channels: {model.input_adapter.in_channels} → {model.output_adapter.out_channels}")
    except Exception as e:
        logger.error(f"Failed to load base model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

    # Validate model has required methods
    required_methods = ['freeze_backbone', 'get_trainable_params', 'compute_losses']
    for method in required_methods:
        if not hasattr(model, method):
            raise AttributeError(f"Model missing required method: {method}")

    # Freeze backbone for parameter-efficient fine-tuning
    try:
        model.freeze_backbone()
        logger.info("Backbone frozen successfully")
    except Exception as e:
        logger.error(f"Failed to freeze backbone: {e}")
        raise RuntimeError(f"Backbone freezing failed: {e}")

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Model moved to device: {device}")

    # Log parameter counts before training
    # This helps track model complexity and training efficiency
    log_parameter_counts(model, logger, wandb_log=wandb_initialized)

    # Prepare dataset and dataloaders
    # Uses the VAE-specific dataloader for optimal data handling
    try:
        train_loader, val_loader = prepare_dataset(args)
        logger.info(f"Created dataloaders with {len(train_loader.dataset)} training and {len(val_loader.dataset)} validation samples")
    except Exception as e:
        logger.error(f"Failed to prepare dataset: {e}")
        raise RuntimeError(f"Dataset preparation failed: {e}")

    # Initialize optimizer with trainable parameters
    # Only adapters are trained, backbone remains frozen
    try:
        trainable_params = model.get_trainable_params()
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
        logger.info(f"Optimizer initialized with {len(trainable_params)} parameter groups")
    except Exception as e:
        logger.error(f"Failed to initialize optimizer: {e}")
        raise RuntimeError(f"Optimizer initialization failed: {e}")

    # Calculate total training steps for scheduler
    # Enables proper learning rate scheduling
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    # Initialize scheduler
    # Cosine schedule with warmup for stable training
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Initialize EMA model
    # Improves training stability and final model quality
    ema_model = EMAModel(model.parameters())

    # Initialize early stopping
    # Prevents overfitting and saves best model
    best_val_loss = float('inf')
    patience_counter = 0
    start_epoch = 0  # Default start epoch

    # Load checkpoint if specified
    if args.resume_from_checkpoint:
        try:
            start_epoch, best_val_loss, patience_counter = load_checkpoint(args.resume_from_checkpoint, model, optimizer, scheduler, device)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise RuntimeError(f"Checkpoint loading failed: {e}")

    # Training loop
    total_epochs_to_train = args.num_epochs - start_epoch
    logger.info(f"Training for {total_epochs_to_train} epochs (from epoch {start_epoch} to {args.num_epochs-1})")
    
    log_interval = 10  # Log every 10 steps (can make configurable)
    current_grad_norm = None  # Track gradient norm for wandb logging
    training_start_time = time.time()
    
    for epoch in range(start_epoch, args.num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_losses = {
            'total_loss': 0,
            'mse_per_channel': torch.zeros(5, device=device),  # 5 bands
            'sam_loss': 0
        }

        # Training phase
        for step, (batch, mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} - Training")):
            batch = batch.to(device)
            mask = mask.to(device) # Background mask (1 for leaf, 0 for background)
            if mask is None:
                logger.warning(f"[MASK WARNING] No mask provided for training batch at step {step}!")
            # Sanitize batch to remove NaNs and Infs, and preserve [-1, 1] range for VAE compatibility
            # Conceptually preserve the NaN-based primary mask for calculating masked losses!
            # Ensures that only valid data is passed to the model, preventing NaN/infinity propagation
            batch = torch.nan_to_num(batch, nan=0.0, posinf=1.0, neginf=-1.0)
            batch = torch.clamp(batch, min=-1.0, max=1.0)  # VAE expects [-1, 1] range

            # Forward pass with mask support for leaf-focused training
            try:
                # Use the model's forward method which now supports mask parameter
                # This automatically handles encoding, decoding, and masked loss computation
                reconstruction, losses = model.forward(sample=batch, mask=mask)

                # Check for NaNs in reconstruction and skip if corrupted
                if torch.isnan(reconstruction).any():
                    logger.warning("NaNs detected in reconstruction — skipping batch")
                    continue

            except Exception as e:
                logger.error(f"Forward pass failed: {e}")
                continue

            # Validate losses structure
            if not isinstance(losses, dict):
                logger.error(f"Model forward() returned non-dict losses: {type(losses)}")
                continue

            # Calculate total loss with proper validation
            # Combines MSE and optional SAM loss
            if 'mse' not in losses:
                logger.error("MSE loss not found in model output")
                continue

            total_loss = losses['mse']
            if args.use_sam_loss and 'sam' in losses:
                total_loss = total_loss + args.sam_weight * losses['sam']
            # Saturation penalty encourages outputs to remain well within the [-1, 1] range.
            # Penalizing values close to ±1 avoids saturating activations which can destroy fine spectral distinctions.
            # This term is optional and controlled by --use_saturation_penalty.
            if 'saturation_penalty' in losses:
                total_loss += losses['saturation_penalty']

            # --- Coordinated Output Range Control System ---
            # This system coordinates two complementary penalties to achieve optimal output range control
            # while preserving spectral fidelity and enabling SD3 compatibility.
            #
            # COORDINATION STRATEGY:
            # 1. Saturation Penalty (in model): Prevents spectral compression by discouraging values near ±1
            #    - Threshold: args.saturation_threshold (default: 0.95, avoids hard saturation that destroys spectral details)
            #    - Weight: args.saturation_penalty_weight (default: 0.05, gentle, preserves spectral relationships)
            #    - Location: Model level (applied during loss computation)
            #
            # 2. Range Penalty (in training): Enforces overall output range for SD3 compatibility
            #    - Threshold: args.range_threshold (default: 1.0, enforces [-1,1] range for downstream pipelines)
            #    - Weight: args.range_penalty_weight (default: 0.2, stronger than saturation, provides output control)
            #    - Location: Training level (applied during optimization)
            #
            # RATIONALE FOR COORDINATION:
            # - Saturation penalty alone: Great for spectral fidelity but insufficient for output range control
            # - Range penalty alone: Good for output range but can cause spectral compression
            # - Combined approach: Spectral fidelity + output range control + SD3 compatibility
            #
            # TRAINING BENEFITS:
            # - Early training: Range penalty quickly brings outputs into reasonable range
            # - Mid training: Saturation penalty prevents spectral compression as outputs approach ±1
            # - Late training: Both penalties work together for optimal spectral fidelity within [-1,1]
            #
            # SPECTRAL SCIENCE BENEFITS:
            # - Preserves fine spectral distinctions important for plant health analysis
            # - Maintains interpretable spectral signatures
            # - Enables downstream SD3 integration without range issues
            if reconstruction is not None and args.use_range_penalty:
                range_penalty = torch.mean(torch.relu(torch.abs(reconstruction) - args.range_threshold))
                total_loss += args.range_penalty_weight * range_penalty

            # --- Learned Per-Band Weighting ---
            # Manually increase weight for bands 3 and 5 due to persistent reconstruction errors.
            # Could be replaced with learned weights or adaptive heuristics later.
            # Apply learned per-band weights (simple example: weight poor-performing bands more)
            #if 'mse_per_channel' in losses:
            #    band_weights = torch.tensor([1.0, 1.0, 1.5, 1.0, 1.5], device=device)  # Bands 3 and 5 upweighted
            #    weighted_mse = (losses['mse_per_channel'] * band_weights).mean()
            #    total_loss = weighted_mse + total_loss - losses['mse']  # Replace original MSE
            # disabled to avoid conflicts with upped sam_loss weight, avoiding optimization conflicts

            # Debugging: Check for NaN/Inf in total loss before backward
            if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                logger.warning("Detected NaN/Inf loss; skipping this batch.")
                continue

            # Log decoder output range every log_interval steps
            # NOTE: Tanh() is applied because "if not self.training" in eval mode!!
            if step % log_interval == 0:
                min_val = reconstruction.min().item()
                max_val = reconstruction.max().item()
                logger.info(f"Decoder output range: min={min_val:.4f}, max={max_val:.4f}")
                # Compute and log per-batch output range stats
                batch_output_stats = compute_output_range_stats(reconstruction)
                logger.info(f"Batch Output Range Stats: "
                            f"min={batch_output_stats['global_min']:.4f}, "
                            f"max={batch_output_stats['global_max']:.4f}, "
                            f"mean={batch_output_stats['global_mean']:.4f}, "
                            f"std={batch_output_stats['global_std']:.4f}")

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()

            # Apply gradient clipping
            # Prevents exploding gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.max_grad_norm
            )
            
            # Track gradient norm for wandb logging
            current_grad_norm = grad_norm.item()

            # Log if gradients were clipped
            # Helps monitor training stability
            if grad_norm > args.max_grad_norm:
                logger.info(f"Gradients clipped at epoch {epoch+1}, norm: {grad_norm:.2f}")

            optimizer.step()
            scheduler.step()

            # Update EMA model
            # Improves model stability
            ema_model.step(model.parameters())
            
            # Log training progress to wandb
            if wandb_initialized:
                log_training_progress_to_wandb(step, total_loss, scheduler.get_last_lr()[0], current_grad_norm, log_interval)

            # Track losses and mask statistics
            # Monitors training progress and mask coverage
            # TODO: dynamic data_range adjustment based on output_range_stats
            # (assumes [-1, 1] range – might produce incorrect SSIM scores if output range differs)
            train_losses['total_loss'] += total_loss.item()
            if 'mse_per_channel' in losses:
                train_losses['mse_per_channel'] += losses['mse_per_channel'].detach()
            if args.use_sam_loss and 'sam' in losses:
                train_losses['sam_loss'] += losses['sam'].item()

        # Validation phase
        model.eval()
        val_losses = {
            'total_loss': 0,
            'mse_per_channel': torch.zeros(5, device=device),  # 5 bands
            'sam_loss': 0
        }

        with torch.no_grad():
            ssim_per_band = []  # To accumulate average SSIM per band over all batches
            output_range_stats = None  # To collect output range statistics
            for batch, mask in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} - Validation"):
                batch = batch.to(device)
                mask = mask.to(device) # Background mask (1 for leaf, 0 for background)
                if mask is None:
                    logger.warning(f"[MASK WARNING] No mask provided for validation batch!")
                # (Removed debug logging for NaNs in validation batch)
                # Sanitize batch to remove NaNs and Infs, and preserve [-1, 1] range for VAE compatibility
                batch = torch.nan_to_num(batch, nan=0.0, posinf=1.0, neginf=-1.0)
                batch = torch.clamp(batch, min=-1.0, max=1.0)  # VAE expects [-1, 1] range

                # Forward pass with mask support for validation
                try:
                    # Use the model's forward method which now supports mask parameter
                    # This automatically handles encoding, decoding, and masked loss computation
                    reconstruction, losses = model.forward(sample=batch, mask=mask)
                    
                    # Check for NaNs in reconstruction and skip if corrupted
                    if torch.isnan(reconstruction).any():
                        logger.warning("NaNs detected in reconstruction — skipping validation batch")
                        continue
                        
                except Exception as e:
                    logger.error(f"Validation forward pass failed: {e}")
                    continue

                # Pass the mask into compute_losses to handle background masking internally.
                # This ensures only leaf regions (mask == 1) contribute to loss, and avoids double-masking issues
                # such as background bounding box artifacts. Background is optionally included via a soft penalty.
                # Validate losses structure
                if not isinstance(losses, dict):
                    logger.error(f"Model forward() returned non-dict losses: {type(losses)}")
                    continue

                # Calculate total loss
                # Consistent with training loss computation
                if 'mse' not in losses:
                    continue

                total_loss = losses['mse']
                if args.use_sam_loss and 'sam' in losses:
                    total_loss = total_loss + args.sam_weight * losses['sam']
                # Debugging: Check for NaN in total loss (validation)
                if torch.isnan(total_loss):
                    logger.warning("Total loss is NaN after MSE and SAM loss aggregation")

                # Compute SSIM per-band (compatibility with skimage.metrics.ssim)
                # Loop over both batch and channel dimensions using skimage.metrics.structural_similarity
                # CRITICAL: Apply mask to both original and reconstructed images before SSIM computation
                # This ensures we only evaluate reconstruction quality on leaf regions, not background

                # NOTE: fallback of computing SSIM without mask will include background, potentially biasing results.
                batch_np = batch.detach()
                recon_np = reconstruction.detach()
                mask_np = mask.detach()  # Background mask (1 for leaf, 0 for background)
                
                # Accumulate average SSIM per band for this batch
                num_bands = batch_np.shape[1]
                batch_size = batch_np.shape[0]
                batch_ssim_per_band = []
                for band_idx in range(num_bands):
                    band_ssim_total = 0.0
                    valid_samples = 0  # Count samples with valid leaf regions
                    for sample_idx in range(batch_size):
                        # Apply mask to isolate leaf regions only
                        original_band = batch_np[sample_idx, band_idx].cpu().numpy()
                        recon_band = recon_np[sample_idx, band_idx].cpu().numpy()
                        sample_mask = mask_np[sample_idx, 0].cpu().numpy()  # (H, W)
                        
                        # Only compute SSIM if there are valid leaf pixels
                        if np.sum(sample_mask) > 0:
                            # Apply mask to both original and reconstructed bands
                            masked_orig = original_band * sample_mask
                            masked_recon = recon_band * sample_mask
                            
                            # Compute SSIM only on the masked region
                            # NOTE: SSIM computation assumes [-1, 1] range, but adapter output may be unconstrained
                            # The data_range parameter should be adjusted based on actual output range
                            # TODO: Consider dynamic data_range adjustment based on output_range_stats
                            try:
                                # Use fixed data_range for now, but consider dynamic adjustment
                                # data_range = output_range_stats['global_max'] - output_range_stats['global_min'] if output_range_stats else 2.0
                                ssim_val = ssim(masked_orig, masked_recon, data_range=2.0,  # [-1, 1] range = 2.0
                                              mask=sample_mask.astype(bool))
                            except TypeError:
                                # Fallback: compute SSIM on entire image but this includes background
                                # This is less accurate but ensures compatibility
                                logger.warning(f"SSIM mask parameter not supported, computing on entire image for sample {sample_idx}")
                                ssim_val = ssim(original_band, recon_band, data_range=2.0)  # [-1, 1] range = 2.0
                            
                            band_ssim_total += ssim_val
                            valid_samples += 1
                    
                    # Average SSIM for this band across valid samples
                    if valid_samples > 0:
                        batch_ssim_per_band.append(band_ssim_total / valid_samples)
                    else:
                        batch_ssim_per_band.append(0.0)  # No valid leaf regions
                
                # Accumulate per-batch SSIMs for averaging after all batches
                if len(ssim_per_band) == 0:
                    ssim_per_band = batch_ssim_per_band
                else:
                    ssim_per_band = [x + y for x, y in zip(ssim_per_band, batch_ssim_per_band)]

                # Track losses and mask statistics
                # Monitors validation performance and mask coverage
                val_losses['total_loss'] += total_loss.item()
                if 'mse_per_channel' in losses:
                    val_losses['mse_per_channel'] += losses['mse_per_channel']
                if args.use_sam_loss and 'sam' in losses:
                    val_losses['sam_loss'] += losses['sam'].item()
                
                # Collect output range statistics (use first batch for efficiency)
                if output_range_stats is None:
                    output_range_stats = compute_output_range_stats(reconstruction)
                    
            # After validation loop, average ssim_per_band over number of batches
            if len(val_loader) > 0:
                ssim_per_band = [v / len(val_loader) for v in ssim_per_band]

        # Average losses
        # Computes epoch-level metrics
        for key in train_losses:
            if isinstance(train_losses[key], torch.Tensor):
                train_losses[key] /= len(train_loader)
            else:
                train_losses[key] /= len(train_loader)

        for key in val_losses:
            if isinstance(val_losses[key], torch.Tensor):
                val_losses[key] /= len(val_loader)
            else:
                val_losses[key] /= len(val_loader)

        # Get band importance if using spectral attention
        # Analyzes model's spectral understanding
        band_importance = {}
        if args.use_spectral_attention and hasattr(model, 'input_adapter') and hasattr(model.input_adapter, 'attention'):
            try:
                band_importance = model.input_adapter.attention.get_band_importance()
            except Exception as e:
                logger.warning(f"Failed to get band importance: {e}")

        # Global scale monitoring:
        # The model applies a learnable scalar multiplier (global_scale) to preserve spectral shape across bands.
        # Log its value every epoch to monitor for convergence stability.
        # Biological interpretability and SD3 compatibility are preserved when scale remains in [0.1, 3.0].
        if hasattr(model.output_adapter, 'global_scale'):
            current_scale = model.output_adapter.global_scale.item()
            logger.info(f"[Epoch {epoch+1}] Global output scale: {current_scale:.4f}")
            if current_scale < 0.1 or current_scale > 3.0:
                logger.warning(f"Global scale {current_scale:.4f} out of expected range (0.1–3.0)")

        # print SSIM
        avg_ssim = np.mean(ssim_per_band)
        print(f"Avg SSIM: {avg_ssim:.4f} | Per-band SSIM: {[f'{v:.4f}' for v in ssim_per_band]}")

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log to training logger
        # Handle tensor conversion properly - multi-element tensors need to be converted to lists
        def convert_tensor_for_logging(v):
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    return v.item()
                else:
                    return v.detach().cpu().tolist()
            return v

        training_logger.log_epoch(
            epoch=epoch,
            train_losses={k: convert_tensor_for_logging(v) for k, v in train_losses.items()},
            val_losses={k: convert_tensor_for_logging(v) for k, v in val_losses.items()},
            band_importance=band_importance,
            ssim_per_band=ssim_per_band,
            global_scale=current_scale,
            learning_rate=scheduler.get_last_lr()[0],
            grad_norm=current_grad_norm,
            output_range_stats=output_range_stats
        )
        
        # Log model health periodically
        if epoch % 5 == 0:  # Every 5 epochs
            health_data = {
                "global_scale": current_scale,
                "memory_usage_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                "grad_norm": current_grad_norm,
                "learning_rate": scheduler.get_last_lr()[0]
            }
            training_logger.log_model_health(health_data)

        # Log metrics
        # Tracks training progress
        log_training_metrics(logger, epoch, train_losses, val_losses, band_importance, args, output_range_stats)
        
        # Save grayscale band 0 of the first sample as PNG for local inspection
        try:
            orig_img = (batch[0][0].detach().cpu().numpy() + 1.0) * 127.5  # from [-1,1] to [0,255]
            recon_img = (reconstruction[0][0].detach().cpu().numpy() + 1.0) * 127.5
            orig_img = np.clip(orig_img, 0, 255).astype(np.uint8)
            recon_img = np.clip(recon_img, 0, 255).astype(np.uint8)
            orig_pil = Image.fromarray(orig_img, mode='L')
            recon_pil = Image.fromarray(recon_img, mode='L')
            orig_pil.save(os.path.join(image_output_dir, f"epoch_{epoch+1}_original_band0.png"))
            recon_pil.save(os.path.join(image_output_dir, f"epoch_{epoch+1}_recon_band0.png"))
        except Exception as e:
            logger.warning(f"Failed to save local PNG images: {e}")
        
        # Log to wandb (includes both metrics and images)
        if wandb_initialized:
            try:
                log_to_wandb(epoch, train_losses, val_losses, band_importance, batch, reconstruction, model, output_range_stats, ssim_per_band, scheduler.get_last_lr()[0], current_grad_norm)
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")

        # Save checkpoint
        # Implements model checkpointing strategy
        is_best = val_losses['total_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_losses['total_loss']
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % args.save_every == 0 or is_best:
            try:
                save_checkpoint(model, optimizer, scheduler, epoch, val_losses['total_loss'], is_best, args, best_val_loss, patience_counter)
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")

        # Early stopping
        # Prevents overfitting
        if patience_counter >= args.early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Save final model
    try:
        final_model_path = os.path.join(args.output_dir, 'final_model')
        model.save_pretrained(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")
    except Exception as e:
        logger.error(f"Failed to save final model: {e}")

    # Log final training summary
    total_training_time = time.time() - training_start_time
    training_logger.log_final_summary(
        total_epochs=epoch + 1,
        total_time=total_training_time,
        final_train_loss=train_losses['total_loss'].item() if isinstance(train_losses['total_loss'], torch.Tensor) else train_losses['total_loss'],
        final_val_loss=val_losses['total_loss'].item() if isinstance(val_losses['total_loss'], torch.Tensor) else val_losses['total_loss'],
        model_path=final_model_path
    )

    # Finish wandb run
    if wandb_initialized and wandb.run:
        wandb.finish()

    logger.info("Training completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Train 5-channel multispectral VAE")
    parser.add_argument("--subfolder", type=str, default="vae",
                        help="Subfolder inside base_model_path where the VAE is stored.")
    parser.add_argument("--train_file_list", type=str, required=True, help="Path to train_files.txt")
    parser.add_argument("--val_file_list", type=str, required=True, help="Path to val_files.txt")
    parser.add_argument("--output_dir", type=str, help="Directory to save model checkpoints (optional)")
    parser.add_argument("--base_model_path", type=str,
                      default="stabilityai/stable-diffusion-3-medium",
                      help="Path to base SD3 model or local VAE checkpoint")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--adapter_placement", type=str, default="both",
                      choices=["input", "output", "both"],
                      help="Where to place adapters")
    parser.add_argument("--use_spectral_attention", action="store_true",
                      help="Use spectral attention mechanism")
    parser.add_argument("--use_sam_loss", action="store_true",
                      help="Use Spectral Angle Mapper loss")
    parser.add_argument("--sam_weight", type=float, default=0.1,
                      help="Weight for SAM loss term")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                      help="Ratio of total steps for warmup")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                      help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                      help="Maximum gradient norm for clipping")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint directory to resume from")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--use_saturation_penalty", action="store_true",
                        help="Enable saturation penalty for outputs nearing [-1, 1] boundaries")
    parser.add_argument("--saturation_penalty_weight", type=float, default=0.05,
                        help="Weight for saturation penalty (default: 0.5 well for spectral fidelity)")
    parser.add_argument("--saturation_threshold", type=float, default=0.95,
                        help="Threshold for saturation penalty (default: 0.95, prevents spectral compression)")
    parser.add_argument("--use_range_penalty", action="store_true",
                        help="Enable range penalty for outputs outside [-1, 1] boundaries")
    parser.add_argument("--range_penalty_weight", type=float, default=0.2,
                        help="Weight for range penalty (default: 0.2er than saturation for output control)")
    parser.add_argument("--range_threshold", type=float, default=1.0,
                        help="Threshold for range penalty (default: 1.0, enforces [-1] output range)")

    args = parser.parse_args()

    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = str(Path(args.train_file_list).parent / "model_output")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)

if __name__ == "__main__":
    main()