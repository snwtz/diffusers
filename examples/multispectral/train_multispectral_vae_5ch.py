"""
Training script for Multispectral VAE Adapter

This script implements the training pipeline for the thesis's core methodological contribution:
a lightweight adapter-based multispectral VAE architecture. It serves as the pretraining step
for integrating a custom VAE into the Stable Diffusion 3 + DreamBooth pipeline for generating
synthetic multispectral plant imagery.

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

3. Training Features:
   a) Data Management:
      - Custom 5-channel MultispectralDataset
      - Deterministic train/val splitting
      - Spectral normalization pipeline
      - Memory-efficient loading
   
   b) Loss Computation:
      - Per-band MSE for spatial fidelity
      - SAM loss for spectral signature preservation
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
        --train_ratio 0.8 \
        --seed 42

    # Then, train the VAE:
    python train_multispectral_vae_5ch.py \
        --output_dir /path/to/save/model \
        --num_epochs 100 \
        --batch_size 8 \
        --learning_rate 1e-4 \
        --adapter_placement both \
        --use_spectral_attention \
        --use_sam_loss \
        --sam_weight 0.1 \
        --warmup_ratio 0.1 \
        --early_stopping_patience 10 \
        --max_grad_norm 1.0
"""

import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
from PIL import Image
import rasterio
from tqdm import tqdm
import logging
import json
from datetime import datetime
import wandb
import shutil
from typing import Tuple

from diffusers import AutoencoderKLMultispectralAdapter
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from multispectral_dataloader import create_multispectral_dataloader
from split_dataset import run_split

class MultispectralDataset(Dataset):
    """Dataset for loading 5-channel multispectral TIFF files from split file lists."""
    
    def __init__(self, file_list_path, transform=None):
        """
        Initialize the dataset.
        
        Args:
            file_list_path: Path to train_files.txt or val_files.txt
            transform: Optional transforms to apply
        """
        self.file_list_path = Path(file_list_path)
        if not self.file_list_path.exists():
            raise FileNotFoundError(f"File list not found: {file_list_path}")
            
        # Read file paths from the list
        with open(self.file_list_path, 'r') as f:
            self.tiff_files = [Path(line.strip()) for line in f.readlines()]
            
        # Validate all files exist
        for file_path in self.tiff_files:
            if not file_path.exists():
                raise FileNotFoundError(f"TIFF file not found: {file_path}")
                
        self.transform = transform
        logging.info(f"Loaded {len(self.tiff_files)} files from {file_list_path}")
        
    def __len__(self):
        return len(self.tiff_files)
    
    def __getitem__(self, idx):
        # Load 5-channel TIFF
        with rasterio.open(self.tiff_files[idx]) as src:
            # Read all 5 bands
            image = src.read()  # Shape: (5, H, W)
            
            # Convert to float and normalize
            image = image.astype(np.float32)
            
            # Apply transforms if any
            if self.transform:
                image = self.transform(image)
            
            return torch.from_numpy(image)

def setup_logging(args):
    # Create logs directory
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

def log_training_metrics(logger, epoch, train_losses, val_losses, band_importance):
    logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
    logger.info("Training Metrics:")
    logger.info(f"  Total Loss: {train_losses['total_loss']:.4f}")
    logger.info("  Per-channel MSE:")
    for i, loss in enumerate(train_losses['mse_per_channel']):
        logger.info(f"    Band {i+1}: {loss:.4f}")
    if 'sam' in train_losses:
        logger.info(f"  SAM Loss: {train_losses['sam']:.4f}")
    
    logger.info("Validation Metrics:")
    logger.info(f"  Total Loss: {val_losses['total_loss']:.4f}")
    logger.info("  Per-channel MSE:")
    for i, loss in enumerate(val_losses['mse_per_channel']):
        logger.info(f"    Band {i+1}: {loss:.4f}")
    if 'sam' in val_losses:
        logger.info(f"  SAM Loss: {val_losses['sam']:.4f}")
    
    logger.info("Band Importance:")
    for band, importance in band_importance.items():
        logger.info(f"  {band}: {importance:.4f}")

def setup_wandb(args):
    wandb.init(
        project="multispectral-vae",
        config={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "adapter_placement": args.adapter_placement,
            "use_spectral_attention": args.use_spectral_attention,
            "use_sam_loss": args.use_sam_loss,
            "warmup_ratio": args.warmup_ratio,
            "early_stopping_patience": args.early_stopping_patience
        }
    )

def log_to_wandb(epoch, train_losses, val_losses, band_importance, batch, reconstruction):
    wandb.log({
        "epoch": epoch,
        "train_total_loss": train_losses['total_loss'],
        "val_total_loss": val_losses['total_loss'],
        **{f"train_mse_band_{i}": loss for i, loss in enumerate(train_losses['mse_per_channel'])},
        **{f"val_mse_band_{i}": loss for i, loss in enumerate(val_losses['mse_per_channel'])},
        **{f"band_importance_{k}": v for k, v in band_importance.items()},
        "original_images": wandb.Image(batch[0]),
        "reconstructed_images": wandb.Image(reconstruction[0])
    })
    if 'sam' in train_losses:
        wandb.log({
            "train_sam_loss": train_losses['sam'],
            "val_sam_loss": val_losses['sam']
        })

def save_checkpoint(model, optimizer, scheduler, epoch, loss, is_best, args):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
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
    """Count trainable and non-trainable parameters in the model."""
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
    
    if wandb_log:
        wandb.log({
            "model/trainable_params": param_counts['trainable'],
            "model/non_trainable_params": param_counts['non_trainable'],
            "model/total_params": param_counts['total'],
            "model/trainable_percentage": param_counts['trainable_percentage']
        })

def prepare_dataset(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare the dataset by using the split files created by split_dataset.py.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger = setup_logging(args)
    
    # Get split files from script directory
    split_dir = Path(__file__).parent
    train_list = split_dir / "train_files.txt"
    val_list = split_dir / "val_files.txt"
    
    # Validate split files exist
    if not train_list.exists() or not val_list.exists():
        raise FileNotFoundError(
            f"Split files not found in {split_dir}. "
            "Please run split_dataset.py first to create train_files.txt and val_files.txt"
        )
    
    # Create datasets using the split files
    train_dataset = MultispectralDataset(
        file_list_path=train_list,
        transform=None  # Add transforms if needed
    )
    val_dataset = MultispectralDataset(
        file_list_path=val_list,
        transform=None  # Add transforms if needed
    )
    
    logger.info(f"Created datasets with {len(train_dataset)} training and {len(val_dataset)} validation samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    return train_loader, val_loader

def train(args: argparse.Namespace) -> None:
    """Main training function."""
    
    # Initialize model with adapter configuration
    model = AutoencoderKLMultispectralAdapter.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        adapter_placement=args.adapter_placement,
        use_spectral_attention=args.use_spectral_attention,
        use_sam_loss=args.use_sam_loss
    )
    
    # Freeze backbone (only adapters will be trained)
    model.freeze_backbone()
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Setup logging
    logger = setup_logging(args)
    
    # Log parameter counts before training
    log_parameter_counts(model, logger)
    
    # Prepare dataset and dataloaders
    train_loader, val_loader = prepare_dataset(args)
    
    # Initialize optimizer with trainable parameters
    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=args.learning_rate)
    
    # Calculate total training steps for scheduler
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # Initialize scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize EMA model
    ema_model = EMAModel(model.parameters())
    
    # Setup wandb
    setup_wandb(args)
    
    # Initialize early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        train_losses = {
            'total_loss': 0,
            'mse_per_channel': torch.zeros(5),  # 5 bands
            'sam_loss': 0
        }
        
        # Training phase
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} - Training"):
            batch = batch.to(device)
            
            # Forward pass
            output = model(batch)
            reconstruction = output.sample
            
            # Get detailed losses from model
            losses = model.compute_losses(batch, reconstruction)
            
            # Calculate total loss
            total_loss = losses['mse']
            if args.use_sam_loss and 'sam' in losses:
                total_loss = total_loss + args.sam_weight * losses['sam']
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Apply gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=args.max_grad_norm
            )
            
            # Log if gradients were clipped
            if grad_norm > args.max_grad_norm:
                logger.info(f"Gradients clipped at epoch {epoch+1}, norm: {grad_norm:.2f}")
                wandb.log({
                    "training/grad_norm": grad_norm,
                    "training/gradients_clipped": 1
                })
            
            optimizer.step()
            scheduler.step()
            
            # Update EMA model
            ema_model.step(model.parameters())
            
            # Track losses
            train_losses['total_loss'] += total_loss.item()
            train_losses['mse_per_channel'] += losses['mse_per_channel'].detach()
            if args.use_sam_loss and 'sam' in losses:
                train_losses['sam_loss'] += losses['sam'].item()
        
        # Validation phase
        model.eval()
        val_losses = {
            'total_loss': 0,
            'mse_per_channel': torch.zeros(5),  # 5 bands
            'sam_loss': 0
        }
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} - Validation"):
                batch = batch.to(device)
                
                # Forward pass
                output = model(batch)
                reconstruction = output.sample
                
                # Get detailed losses from model
                losses = model.compute_losses(batch, reconstruction)
                
                # Calculate total loss
                total_loss = losses['mse']
                if args.use_sam_loss and 'sam' in losses:
                    total_loss = total_loss + args.sam_weight * losses['sam']
                
                # Track losses
                val_losses['total_loss'] += total_loss.item()
                val_losses['mse_per_channel'] += losses['mse_per_channel']
                if args.use_sam_loss and 'sam' in losses:
                    val_losses['sam_loss'] += losses['sam'].item()
        
        # Average losses
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
        band_importance = {}
        if args.use_spectral_attention:
            band_importance = model.input_adapter.attention.get_band_importance()
        
        # Log metrics
        log_training_metrics(logger, epoch, train_losses, val_losses, band_importance)
        log_to_wandb(epoch, train_losses, val_losses, band_importance, batch, reconstruction)
        
        # Save checkpoint
        is_best = val_losses['total_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_losses['total_loss']
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % args.save_every == 0 or is_best:
            save_checkpoint(model, optimizer, scheduler, epoch, val_losses['total_loss'], is_best, args)
        
        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

def main():
    parser = argparse.ArgumentParser(description="Train 5-channel multispectral VAE")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
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
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main() 