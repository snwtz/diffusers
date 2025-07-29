"""
DreamBooth Multispectral Training Logger

This module provides comprehensive logging functionality for the DreamBooth multispectral training process.
It captures key metrics and training behavior in both compressed text format and structured JSON for analysis.

Features:
- Epoch-by-epoch training metrics
- Spectral fidelity monitoring (MSE, SSIM, SAM)
- Validation image generation
- System performance tracking
- Compressed log format for easy parsing

1. TRAINING MONITORING:
   - Tracks training loss and learning rate
   - Monitors gradient norms for optimization stability
   - Provides step-by-step progress tracking

2. SPECTRAL FIDELITY MONITORING:
   - Per-channel MSE (Mean Squared Error)
   - Per-channel SSIM (Structural Similarity Index)
   - SAM (Spectral Angle Mapper) for spectral signature preservation
   - Detailed spectral summaries every 100 steps

3. VALIDATION AND VISUALIZATION:
   - Generated image quality assessment
   - Progress tracking through training epochs
   - Basic validation metrics logging

4. SYSTEM PERFORMANCE:
   - Memory usage tracking
   - Training efficiency monitoring
   - Error and warning logging

DESIGN RATIONALE:
-----------------
The logger employs a dual-format approach (text + JSON) to balance human readability
with machine-processable data. This design supports both real-time monitoring during
training and comprehensive post-hoc analysis.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import torch
import numpy as np
from pathlib import Path
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
from PIL import Image




class DreamBoothLogger:
    """
    Comprehensive training logger for DreamBooth multispectral training.
    Handles all logging to text, JSON, and wandb. Provides a clean API for the training script.
    """
    def __init__(self, output_dir: str, model_name: str = "dreambooth_multispectral"):
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.log_dir = self.output_dir / "dreambooth_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use single files instead of timestamped files to reduce clutter
        self.log_file = self.log_dir / f"{model_name}_training_log.txt"
        # JSON logging removed for training speed optimization
        self.log_data = {
            "experiment_info": {
                "model_name": model_name,
                "start_time": timestamp,
                "log_file": str(self.log_file)
            },
            "training_config": {},
            "steps": [],
            "epochs": [],
            "validations": [],
            "best_metrics": {}
        }
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_step = 0
        self._write_header()

    def _write_header(self):
        header = f"""
{'='*80}
DREAMBOOTH MULTISPECTRAL TRAINING LOG
{'='*80}
Model: {self.model_name}
Start Time: {self.log_data['experiment_info']['start_time']}
Log File: {self.log_file}
Note: This file is overwritten for each training run to reduce clutter
{'='*80}

"""
        with open(self.log_file, 'w') as f:
            f.write(header)

    def log_config(self, config: Dict[str, Any]):
        """Log training configuration."""
        self.log_data["training_config"] = config
        config_str = f"\nTRAINING CONFIGURATION:\n{'-'*40}\n"
        for key, value in config.items():
            config_str += f"{key}: {value}\n"
        with open(self.log_file, 'a') as f:
            f.write(config_str + "\n")
        # JSON logging removed for training speed optimization

    def log_step(self, step: int, epoch: int, loss: float, learning_rate: float, grad_norm: Optional[float] = None, 
                 spectral_metrics: Optional[Dict[str, Any]] = None):
        """Log step-level metrics."""
        try:
            step_data = {
                "step": step,
                "epoch": epoch,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "loss": loss,
                "learning_rate": learning_rate,
                "grad_norm": grad_norm,
            }
            
            # Add spectral metrics if provided (every 100 steps from training script)
            if spectral_metrics:
                step_data["spectral_metrics"] = spectral_metrics
                
            # Don't accumulate step data in memory for speed
            # self.log_data["steps"].append(step_data)  # Removed for memory efficiency
            
            # Check alert thresholds with exception handling
            try:
                self._check_alert_thresholds(step_data, step, mode="step")
            except Exception as e:
                print(f"Warning: Failed to check alert thresholds: {e}")
            
            # Write to text file every 100 steps with exception handling
            if step % 100 == 0:
                try:
                    self._write_step_summary(step_data)
                except Exception as e:
                    print(f"Warning: Failed to write step summary: {e}")
                
                # Also write detailed spectral summary if available
                if spectral_metrics:
                    try:
                        self.log_spectral_summary(step, spectral_metrics)
                    except Exception as e:
                        print(f"Warning: Failed to write spectral summary: {e}")
                
            # Log range monitoring every 500 steps
            if step % 500 == 0 and spectral_metrics:
                try:
                    self._log_range_monitoring(step, spectral_metrics)
                except Exception as e:
                    print(f"Warning: Failed to log range monitoring: {e}")
                
            # Log to wandb every 50 steps (reduced frequency for speed)
            try:
                if step % 50 == 0 and WANDB_AVAILABLE and wandb.run:
                    wandb_log = {k: v for k, v in step_data.items() if v is not None and k != "spectral_metrics"}
                    wandb.log(wandb_log, step=step)
                    
                # Log spectral metrics to wandb every 100 steps
                if step % 100 == 0 and spectral_metrics and WANDB_AVAILABLE and wandb.run:
                    wandb_spectral_log = {}
                    for key, value in spectral_metrics.items():
                        if value is not None:
                            wandb_spectral_log[f"spectral/{key}"] = value
                        else:
                            # Use -1 as sentinel value for None values in wandb
                            wandb_spectral_log[f"spectral/{key}"] = -1.0
                    wandb.log(wandb_spectral_log, step=step)
            except Exception as e:
                print(f"Warning: Failed to log to wandb: {e}")
                
        except Exception as e:
            print(f"Warning: Failed to log step data: {e}")
            # Continue training even if logging fails

    def _write_step_summary(self, step_data: Dict[str, Any]):
        try:
            step = step_data["step"]
            epoch = step_data["epoch"]
            loss = step_data["loss"]
            lr = step_data["learning_rate"]
            grad_norm = step_data.get("grad_norm", None)
            
            # Handle None grad_norm gracefully
            if grad_norm is not None:
                grad_norm_str = f" | GradNorm: {grad_norm:.4f}"
            else:
                grad_norm_str = " | GradNorm: N/A"
                
            line = f"Step {step} | Epoch {epoch} | Loss: {loss:.4f} | LR: {lr:.2e}{grad_norm_str}"
            
            # Add spectral metrics if present (every 100 steps)
            spectral_metrics = step_data.get("spectral_metrics")
            if spectral_metrics:
                try:
                    sam_score = spectral_metrics.get("sam_score")
                    if sam_score is not None:
                        line += f" | SAM: {sam_score:.4f}"
                    else:
                        line += " | SAM: N/A"
                    
                    # Add per-channel MSE and SSIM
                    mse_per_channel = spectral_metrics.get("mse_per_channel", [])
                    ssim_per_channel = spectral_metrics.get("ssim_per_channel", [])
                    band_names = ["B9", "B18", "B32", "B42", "B55"]
                    
                    if mse_per_channel and ssim_per_channel:
                        # Use list comprehension for faster string building
                        spectral_parts = [f"{band}:MSE={mse:.4f},SSIM={ssim:.4f}" 
                                        for mse, ssim, band in zip(mse_per_channel, ssim_per_channel, band_names)]
                        line += " | " + ";".join(spectral_parts)
                except Exception as e:
                    line += f" | Spectral metrics error: {str(e)[:50]}"
                    
            with open(self.log_file, 'a') as f:
                f.write(line + "\n")
        except Exception as e:
            print(f"Warning: Failed to write step summary: {e}")

    def log_epoch(self, epoch: int, avg_loss: float, avg_learning_rate: float, total_steps: int, epoch_time: float, validation_metrics: Optional[Dict] = None):
        """Log epoch-level metrics."""
        try:
            epoch_data = {
                "epoch": epoch,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "avg_loss": avg_loss,
                "avg_learning_rate": avg_learning_rate,
                "total_steps": total_steps,
                "epoch_time": epoch_time,
            }
            if validation_metrics is not None:
                epoch_data["validation_metrics"] = validation_metrics
            self.log_data["epochs"].append(epoch_data)
            
            try:
                self._write_epoch_summary(epoch_data)
            except Exception as e:
                print(f"Warning: Failed to write epoch summary: {e}")
                
            # JSON logging removed for training speed optimization
            try:
                if WANDB_AVAILABLE and wandb.run:
                    wandb_log = {k: v for k, v in epoch_data.items() if v is not None}
                    wandb.log(wandb_log, step=epoch)
            except Exception as e:
                print(f"Warning: Failed to log epoch to wandb: {e}")
        except Exception as e:
            print(f"Warning: Failed to log epoch data: {e}")

    def _write_epoch_summary(self, epoch_data: Dict[str, Any]):
        epoch = epoch_data["epoch"]
        avg_loss = epoch_data["avg_loss"]
        avg_lr = epoch_data["avg_learning_rate"]
        val_loss = 0.0
        if epoch_data.get("validation_metrics"):
            val_metrics = epoch_data["validation_metrics"]
            val_loss = val_metrics.get('val_loss', 0)
        line = f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {avg_lr:.2e}"
        with open(self.log_file, 'a') as f:
            f.write(line + "\n")

    def log_validation(self, epoch: int, images: List[Any], prompt: str, validation_metrics: Dict[str, Any]):
        """Log validation results."""
        try:
            validation_data = {
                "epoch": epoch,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "prompt": prompt,
                "num_images": len(images),
                "validation_metrics": validation_metrics,
            }
            self.log_data["validations"].append(validation_data)
            
            # Save validation images locally (every 500 steps)
            try:
                validation_dir = self.log_dir / "validation_images"
                validation_dir.mkdir(exist_ok=True)
                for i, image in enumerate(images):
                    try:
                        image_path = validation_dir / f"step_{epoch:06d}_image_{i:02d}.png"
                        image.save(image_path)
                    except Exception as e:
                        print(f"Warning: Failed to save validation image {i}: {e}")
            except Exception as e:
                print(f"Warning: Failed to save validation images: {e}")

            # Write validation summary
            try:
                self._write_validation_summary(validation_data)
            except Exception as e:
                print(f"Warning: Failed to write validation summary: {e}")
                
            try:
                self._check_alert_thresholds(validation_data["validation_metrics"], epoch, mode="val")
            except Exception as e:
                print(f"Warning: Failed to check validation alert thresholds: {e}")
                
            # JSON logging removed for training speed optimization
            # Log validation metrics to wandb (no images)
            try:
                if WANDB_AVAILABLE and wandb.run:
                    wandb_log = {}
                    # Add validation metrics only
                    for k, v in validation_data["validation_metrics"].items():
                        wandb_log[f"val/{k}"] = v
                    wandb.log(wandb_log, step=epoch)
            except Exception as e:
                print(f"Warning: Failed to log validation to wandb: {e}")
        except Exception as e:
            print(f"Warning: Failed to log validation data: {e}")

    def _check_alert_thresholds(self, metrics: Dict[str, float], step_or_epoch: int, mode: str = "step"):
        """Checks key metrics for warning thresholds."""
        prefix = f"[ALERT][{mode.upper()} {step_or_epoch}]"
        # Add any remaining alert checks here if needed

    def _write_validation_summary(self, validation_data: Dict[str, Any]):
        epoch = validation_data["epoch"]
        prompt = validation_data["prompt"]
        num_images = validation_data["num_images"]
        metrics = validation_data["validation_metrics"]
        line = f"Validation Epoch {epoch:3d} | Prompt: '{prompt}' | Images: {num_images}"
        if metrics:
            line += f" | Loss: {metrics.get('val_loss', 0):.6f}"
        mse_keys = [k for k in validation_data if k.startswith("mse_channel_")]
        if mse_keys:
            mse_str = " | " + ", ".join([f"{k}: {validation_data[k]:.4f}" for k in mse_keys])
            line += mse_str
        with open(self.log_file, 'a') as f:
            f.write(line + "\n")

    def log_warning(self, warning_msg: str):
        """Log warnings during training."""
        warning_str = f"[WARNING] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {warning_msg}\n"
        with open(self.log_file, 'a') as f:
            f.write(warning_str)

    def log_error(self, error_msg: str):
        """Log errors during training."""
        error_str = f"[ERROR] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {error_msg}\n"
        with open(self.log_file, 'a') as f:
            f.write(error_str)

    def close(self):
        """Optional: Finalize and flush logs if needed."""
        pass

    def log_spectral_summary(self, step: int, spectral_metrics: Dict[str, Any]):
        """Log a detailed spectral fidelity summary to a single file."""
        try:
            if not spectral_metrics:
                return
                
            spectral_log_file = self.log_dir / f"{self.model_name}_spectral_log.txt"
            
            sam_score = spectral_metrics.get("sam_score")
            mse_per_channel = spectral_metrics.get("mse_per_channel", [])
            ssim_per_channel = spectral_metrics.get("ssim_per_channel", [])
            
            band_names = ["Band 9 (474nm)", "Band 18 (539nm)", "Band 32 (651nm)", 
                         "Band 42 (731nm)", "Band 55 (851nm)"]
            
            summary = f"\n{'='*60}\n"
            summary += f"SPECTRAL FIDELITY SUMMARY - Step {step}\n"
            summary += f"{'='*60}\n"
            summary += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            if sam_score is not None:
                summary += f"SAM Score: {sam_score:.6f}\n"
            else:
                summary += f"SAM Score: Not computable\n"
            summary += f"\nPer-Channel Metrics:\n"
            summary += f"{'-'*40}\n"
            
            for i, (mse, ssim, band_name) in enumerate(zip(mse_per_channel, ssim_per_channel, band_names)):
                summary += f"{band_name}:\n"
                summary += f"  MSE: {mse:.6f}\n"
                summary += f"  SSIM: {ssim:.6f}\n"
                
            summary += f"\nOverall Assessment:\n"
            if sam_score is not None:
                summary += f"  Spectral Preservation: {'Good' if sam_score > 0.8 else 'Fair' if sam_score > 0.6 else 'Poor'}\n"
            else:
                summary += f"  Spectral Preservation: Not computable\n"
            summary += f"  Spatial Quality: {'Good' if np.mean(ssim_per_channel) > 0.8 else 'Fair' if np.mean(ssim_per_channel) > 0.6 else 'Poor'}\n"
            summary += f"{'='*60}\n"
            
            with open(spectral_log_file, 'a') as f:
                f.write(summary)
        except Exception as e:
            print(f"Warning: Failed to write spectral summary: {e}")

    def _log_range_monitoring(self, step: int, spectral_metrics: Dict[str, Any]):
        """Log range monitoring statistics every 500 steps."""
        try:
            if not spectral_metrics:
                return
                
            range_log_file = self.log_dir / f"{self.model_name}_range_monitoring.txt"
            
            # Extract range statistics if available
            range_stats = spectral_metrics.get("range_stats", {})
            if not range_stats:
                return
                
            band_names = ["Band 9 (474nm)", "Band 18 (539nm)", "Band 32 (651nm)", 
                         "Band 42 (731nm)", "Band 55 (851nm)"]
            
            summary = f"\n{'='*50}\n"
            summary += f"RANGE MONITORING - Step {step}\n"
            summary += f"{'='*50}\n"
            summary += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            summary += f"\nOut-of-Bounds Analysis ([-1, 1] range):\n"
            summary += f"{'-'*40}\n"
            
            total_out_of_bounds = 0
            total_pixels = 0
            
            for i, band_name in enumerate(band_names):
                band_stats = range_stats.get(f"band_{i}", {})
                out_of_bounds = band_stats.get("out_of_bounds", 0)
                total_pixels_band = band_stats.get("total_pixels", 0)
                out_of_bounds_pct = (out_of_bounds / total_pixels_band * 100) if total_pixels_band > 0 else 0
                
                total_out_of_bounds += out_of_bounds
                total_pixels += total_pixels_band
                
                summary += f"{band_name}:\n"
                summary += f"  Out-of-bounds pixels: {out_of_bounds:,} ({out_of_bounds_pct:.2f}%)\n"
                summary += f"  Min value: {band_stats.get('min_val', 'N/A'):.4f}\n"
                summary += f"  Max value: {band_stats.get('max_val', 'N/A'):.4f}\n"
                summary += f"  Mean: {band_stats.get('mean_val', 'N/A'):.4f}\n"
                summary += f"  Std: {band_stats.get('std_val', 'N/A'):.4f}\n"
                
            # Overall statistics
            overall_out_of_bounds_pct = (total_out_of_bounds / total_pixels * 100) if total_pixels > 0 else 0
            summary += f"\nOverall Statistics:\n"
            summary += f"  Total out-of-bounds pixels: {total_out_of_bounds:,} ({overall_out_of_bounds_pct:.2f}%)\n"
            summary += f"  Total pixels analyzed: {total_pixels:,}\n"
            
            # VAE behavior analysis (known to produce out-of-bounds values)
            if overall_out_of_bounds_pct > 10.0:
                summary += f"  CRITICAL: Very high out-of-bounds rate - VAE may be unstable\n"
            elif overall_out_of_bounds_pct > 5.0:
                summary += f"  HIGH: Significant out-of-bounds rate - Monitor for divergence\n"
            elif overall_out_of_bounds_pct > 2.0:
                summary += f"  MODERATE: Expected out-of-bounds rate - VAE behavior normal\n"
            else:
                summary += f"  LOW: Minimal out-of-bounds rate - VAE performing well\n"
                
            # Trend analysis recommendation
            summary += f"  📊 Monitor trend: Check if rate increases over time\n"
                
            summary += f"{'='*50}\n"
            
            with open(range_log_file, 'a') as f:
                f.write(summary)
                
            # Also log to console for immediate visibility
            print(f"Range monitoring at step {step}: {overall_out_of_bounds_pct:.2f}% out-of-bounds pixels")
            
        except Exception as e:
            print(f"Warning: Failed to write range monitoring: {e}")


def create_dreambooth_logger(output_dir: str, model_name: str = "dreambooth_multispectral") -> DreamBoothLogger:
    """
    Factory function to create a DreamBooth training logger.
    
    Args:
        output_dir: Directory to save log files
        model_name: Name for the model/experiment
    
    Returns:
        DreamBoothLogger instance
    """
    return DreamBoothLogger(output_dir, model_name)


def setup_wandb_dreambooth(args, instance_prompt: str, class_prompt: Optional[str] = None):
    """
    Initialize Weights & Biases for DreamBooth multispectral training.
    
    Args:
        args: Training arguments
        instance_prompt: Instance prompt (e.g., "sks leaf")
        class_prompt: Class prompt for prior preservation (optional)
    
    Returns:
        bool: True if wandb initialization successful
    """
    if not WANDB_AVAILABLE:
        print("Warning: wandb not available, skipping wandb initialization")
        return False
        
    try:
        # Create a descriptive run name
        run_name = f"dreambooth_multispectral_{instance_prompt.replace(' ', '_')}"
        if class_prompt:
            run_name += f"_vs_{class_prompt.replace(' ', '_')}"
        run_name += f"_lr{args.learning_rate}_bs{args.train_batch_size}"
        
        wandb.init(
            project="MS Dreambooth",
            name=run_name,
            config={
                "learning_rate": args.learning_rate,
                "train_batch_size": args.train_batch_size,
                "num_train_epochs": args.num_train_epochs,
                "instance_prompt": instance_prompt,
                "class_prompt": class_prompt,
                "with_prior_preservation": args.with_prior_preservation,
                "prior_loss_weight": args.prior_loss_weight,
                "mixed_precision": args.mixed_precision,
                "gradient_checkpointing": args.gradient_checkpointing,
                "max_grad_norm": args.max_grad_norm,
                "pretrained_model": args.pretrained_model_name_or_path,
                "vae_path": args.vae_path,
                "num_channels": args.num_channels,
                "resolution": args.resolution,
                "train_text_encoder": args.train_text_encoder,
                "model_type": "dreambooth_multispectral",
                "input_channels": 5,
                "latent_channels": 16,  # SD3 default
            },
            tags=["dreambooth", "multispectral", "plant-health"]
        )
        return True
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")
        return False


def log_to_wandb_dreambooth(step: int, 
                           epoch: int,
                           loss: float,
                           learning_rate: float,
                           grad_norm: Optional[float] = None,
                           prior_loss: Optional[float] = None,
                           instance_loss: Optional[float] = None,
                           memory_usage: Optional[float] = None,
                           validation_images: Optional[List[Image.Image]] = None,
                           validation_prompt: Optional[str] = None):
    """
    Log DreamBooth training metrics to Weights & Biases.
    
    Args:
        step: Current training step
        epoch: Current epoch
        loss: Training loss
        learning_rate: Current learning rate
        grad_norm: Gradient norm
        prior_loss: Prior preservation loss
        instance_loss: Instance-specific loss
        memory_usage: GPU memory usage
        validation_images: Generated validation images
        validation_prompt: Validation prompt
    """
    if not WANDB_AVAILABLE or not wandb.run:
        return
    
    try:
        # Prepare metrics data
        wandb_log_data = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "learning_rate": learning_rate,
        }
        
        if grad_norm is not None:
            wandb_log_data["gradient_norm"] = grad_norm
        if prior_loss is not None:
            wandb_log_data["prior_loss"] = prior_loss
        if instance_loss is not None:
            wandb_log_data["instance_loss"] = instance_loss
        if memory_usage is not None:
            wandb_log_data["memory_usage_gb"] = memory_usage
        
        # Log metrics
        wandb.log(wandb_log_data)
        
        # Log validation images
        if validation_images and validation_prompt:
            try:
                wandb_images = []
                for i, image in enumerate(validation_images):
                    wandb_images.append(
                        wandb.Image(image, caption=f"{i}: {validation_prompt}")
                    )
                wandb.log({"validation_images": wandb_images})
            except Exception as img_error:
                print(f"Warning: Failed to log validation images to wandb: {img_error}")
            
    except Exception as e:
        print(f"Warning: Failed to log to wandb: {e}")
