"""
DreamBooth Multispectral Training Logger

TODO in eval:
-  all MSEs in one panel

This module provides comprehensive logging functionality for the DreamBooth multispectral training process.
It captures key metrics, concept learning progress, spectral fidelity, and training behavior in both
compressed text format and structured JSON for analysis.

Features:
- Epoch-by-epoch training metrics
- Concept learning analysis
- Spectral fidelity monitoring
- Validation image generation
- System performance tracking
- Compressed log format for easy parsing

1. CONCEPT LEARNING MONITORING:
   - Tracks how well the model learns the "sks" concept
   - Monitors prior preservation loss for concept stability
   - Analyzes text encoder adaptation to multispectral concepts
   - Provides interpretable insights into concept-spectral mapping

2. SPECTRAL FIDELITY ASSESSMENT:
   - Monitors VAE latent space quality during training
   - Tracks spectral signature preservation
   - Analyzes multispectral-to-RGB conversion quality
   - Enables detection of spectral information loss

3. TRAINING STABILITY ANALYSIS:
   - Gradient clipping detection for optimization stability
   - Learning rate scheduling effectiveness
   - Mixed precision training performance
   - Memory usage and computational efficiency

4. VALIDATION AND VISUALIZATION:
   - Generated image quality assessment
   - Spectral concept visualization
   - Comparison with original multispectral data
   - Progress tracking through training epochs

DESIGN RATIONALE:
-----------------
The logger employs a dual-format approach (text + JSON) to balance human readability
with machine-processable data. This design supports both real-time monitoring during
training and comprehensive post-hoc analysis for scientific publication.

The DreamBooth-specific metrics address unique challenges in concept learning:
- Concept preservation vs. spectral fidelity trade-offs
- Text-to-spectral concept mapping
- Prior preservation effectiveness
- Multispectral generation quality
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import torch
import numpy as np
from pathlib import Path
import math
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    ssim = None

# --- Helper functions for post-adapter stats, latent stats, and SSIM ---
def compute_post_adapter_range_stats(decoded_imgs):
    stats = {}
    stats["min"] = decoded_imgs.min().item()
    stats["max"] = decoded_imgs.max().item()
    stats["mean"] = decoded_imgs.mean().item()
    stats["std"] = decoded_imgs.std().item()
    clipped = (decoded_imgs < -1.0) | (decoded_imgs > 1.0)
    stats["percent_clipped"] = clipped.sum().item() / decoded_imgs.numel() * 100
    return stats

def compute_per_band_ssim(pred, target, mask):
    if not SSIM_AVAILABLE:
        return [0.0] * pred.shape[1]  # Return zeros if SSIM not available
        
    ssim_per_band = []
    for i in range(pred.shape[1]):
        pred_band = pred[:, i, :, :].squeeze().cpu().numpy()
        tgt_band = target[:, i, :, :].squeeze().cpu().numpy()
        msk_band = mask.squeeze().cpu().numpy().astype(bool)
        if msk_band.sum() == 0:
            ssim_score = 0.0
        else:
            pred_band = pred_band * msk_band
            tgt_band = tgt_band * msk_band
            ssim_score = ssim(tgt_band, pred_band, data_range=2.0)
        ssim_per_band.append(ssim_score)
    return ssim_per_band


class DreamBoothLogger:
    """
    Comprehensive training logger for DreamBooth multispectral training.
    Handles all logging to text, JSON, and wandb. Provides a clean API for the training script.
    """
    def __init__(self, output_dir: str, model_name: str = "dreambooth_multispectral", vae=None):
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.log_dir = self.output_dir / "dreambooth_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{model_name}_training_log_{timestamp}.txt"
        self.json_file = self.log_dir / f"{model_name}_training_log_{timestamp}.json"
        self.log_data = {
            "experiment_info": {
                "model_name": model_name,
                "start_time": timestamp,
                "log_file": str(self.log_file),
                "json_file": str(self.json_file)
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
        self.vae = vae
        self._write_header()

    def _write_header(self):
        header = f"""
{'='*80}
DREAMBOOTH MULTISPECTRAL TRAINING LOG
{'='*80}
Model: {self.model_name}
Start Time: {self.log_data['experiment_info']['start_time']}
Log File: {self.log_file}
JSON File: {self.json_file}
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
        with open(self.json_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)

    def log_step(self, step: int, epoch: int, loss: float, learning_rate: float, grad_norm: Optional[float] = None, mse_per_channel: Optional[Any] = None, sam_loss: Optional[float] = None, mse_per_channel_global: Optional[Any] = None, sam_loss_global: Optional[float] = None):
        """Log step-level metrics, including per-channel MSE and SAM loss (both mask-only and global if provided)."""
        step_data = {
            "step": step,
            "epoch": epoch,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "loss": loss,
            "learning_rate": learning_rate,
            "grad_norm": grad_norm,
        }
        # Handle per-channel MSE (mask-only)
        if mse_per_channel is not None:
            if hasattr(mse_per_channel, 'detach'):
                mse_per_channel = mse_per_channel.detach().cpu().tolist()
            for idx, val in enumerate(mse_per_channel):
                step_data[f"mse_channel_{idx}_mask"] = float(val)
        # Handle per-channel MSE (global)
        if mse_per_channel_global is not None:
            if hasattr(mse_per_channel_global, 'detach'):
                mse_per_channel_global = mse_per_channel_global.detach().cpu().tolist()
            for idx, val in enumerate(mse_per_channel_global):
                step_data[f"mse_channel_{idx}_global"] = float(val)
        # Handle SAM loss (mask-only)
        if sam_loss is not None:
            step_data["sam_loss_rad_mask"] = float(sam_loss)
            step_data["sam_loss_deg_mask"] = float(sam_loss * 180.0 / math.pi)
        # Handle SAM loss (global)
        if sam_loss_global is not None:
            step_data["sam_loss_rad_global"] = float(sam_loss_global)
            step_data["sam_loss_deg_global"] = float(sam_loss_global * 180.0 / math.pi)
        self.log_data["steps"].append(step_data)
        self._check_alert_thresholds(step_data, step, mode="step")
        # Write to text file every 100 steps
        if step % 100 == 0:
            self._write_step_summary(step_data)
        # Log to wandb every 10 steps
        if step % 10 == 0 and WANDB_AVAILABLE and wandb.run:
            wandb_log = {k: v for k, v in step_data.items() if v is not None}
            wandb.log(wandb_log, step=step)
        # Save JSON backup
        with open(self.json_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)

    def _write_step_summary(self, step_data: Dict[str, Any]):
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
        # Add per-channel MSE if present
        mse_keys = [k for k in step_data if k.startswith("mse_channel_")]
        if mse_keys:
            mse_str = " | " + ", ".join([f"{k}: {step_data[k]:.4f}" for k in mse_keys])
            line += mse_str
        with open(self.log_file, 'a') as f:
            f.write(line + "\n")

    def log_epoch(self, epoch: int, avg_loss: float, avg_learning_rate: float, total_steps: int, epoch_time: float, validation_metrics: Optional[Dict] = None):
        """Log epoch-level metrics."""
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
        self._write_epoch_summary(epoch_data)
        with open(self.json_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
        if WANDB_AVAILABLE and wandb.run:
            wandb_log = {k: v for k, v in epoch_data.items() if v is not None}
            wandb.log(wandb_log, step=epoch)

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

    def log_validation(self, epoch: int, images: List[Any], prompt: str, validation_metrics: Dict[str, Any], mse_per_channel: Optional[Any] = None, sam_loss: Optional[float] = None, mse_per_channel_global: Optional[Any] = None, sam_loss_global: Optional[float] = None):
        """Log validation results, including images, per-channel MSE, and SAM loss (both mask-only and global if provided)."""
        validation_data = {
            "epoch": epoch,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompt,
            "num_images": len(images),
            "validation_metrics": validation_metrics,
        }
        # Handle per-channel MSE (mask-only)
        if mse_per_channel is not None:
            if hasattr(mse_per_channel, 'detach'):
                mse_per_channel = mse_per_channel.detach().cpu().tolist()
            for idx, val in enumerate(mse_per_channel):
                validation_data[f"mse_channel_{idx}_mask"] = float(val)
        # Handle per-channel MSE (global)
        if mse_per_channel_global is not None:
            if hasattr(mse_per_channel_global, 'detach'):
                mse_per_channel_global = mse_per_channel_global.detach().cpu().tolist()
            for idx, val in enumerate(mse_per_channel_global):
                validation_data[f"mse_channel_{idx}_global"] = float(val)
        # Handle SAM loss (mask-only)
        if sam_loss is not None:
            validation_data["sam_loss_rad_mask"] = float(sam_loss)
            validation_data["sam_loss_deg_mask"] = float(sam_loss * 180.0 / math.pi)
        # Handle SAM loss (global)
        if sam_loss_global is not None:
            validation_data["sam_loss_rad_global"] = float(sam_loss_global)
            validation_data["sam_loss_deg_global"] = float(sam_loss_global * 180.0 / math.pi)
        self.log_data["validations"].append(validation_data)
        # Save validation images
        validation_dir = self.log_dir / "validation_images"
        validation_dir.mkdir(exist_ok=True)
        for i, image in enumerate(images):
            image_path = validation_dir / f"epoch_{epoch:03d}_image_{i:02d}.png"
            image.save(image_path)

        # --- VAE decoding and metric computation ---
        if self.vae is not None:
            try:
                with torch.no_grad():
                    # Expect images[i] contains {"latent": Tensor, "target": Tensor, "mask": Tensor}
                    decoded_imgs = self.vae.decode(torch.stack([i["latent"] for i in images[:3]]))
                    val_images = torch.stack([i["target"] for i in images[:3]])
                    val_mask = torch.stack([i["mask"] for i in images[:3]])

                # Clamp decoded images for output comparison
                decoded_imgs_clamped = torch.clamp(decoded_imgs, -1.0, 1.0)

                # Compute post-adapter range stats
                range_stats = compute_post_adapter_range_stats(decoded_imgs)
                # Track clamped output stats for comparison
                range_stats_clamped = compute_post_adapter_range_stats(decoded_imgs_clamped)
                for k, v in range_stats.items():
                    validation_data["validation_metrics"][f"range/{k}"] = v
                for k, v in range_stats_clamped.items():
                    validation_data["validation_metrics"][f"range_clamped/{k}"] = v

                # Compute per-band SSIM (mask-only)
                if SSIM_AVAILABLE:
                    ssim_per_band_mask = compute_per_band_ssim(decoded_imgs, val_images, val_mask)
                    for i, score in enumerate(ssim_per_band_mask):
                        validation_data["validation_metrics"][f"ssim_channel_{i}_mask"] = score
                else:
                    for i in range(decoded_imgs.shape[1]):
                        validation_data["validation_metrics"][f"ssim_channel_{i}_mask"] = 0.0

                # Compute per-band SSIM (global)
                if SSIM_AVAILABLE:
                    ones_mask = torch.ones_like(val_mask)
                    ssim_per_band_global = compute_per_band_ssim(decoded_imgs, val_images, ones_mask)
                    for i, score in enumerate(ssim_per_band_global):
                        validation_data["validation_metrics"][f"ssim_channel_{i}_global"] = score
                else:
                    for i in range(decoded_imgs.shape[1]):
                        validation_data["validation_metrics"][f"ssim_channel_{i}_global"] = 0.0

                # Optionally compute latent stats (if available)
                if "latent" in images[0]:
                    latents = torch.stack([i["latent"] for i in images[:3]])
                    latent_stats = compute_latent_statistics(latents)
                    for k, v in latent_stats.items():
                        validation_data["validation_metrics"][f"latent/{k}"] = v

                # Compute SAM on all 5 channels (mask-only)
                pred_vectors = decoded_imgs.permute(0, 2, 3, 1).reshape(-1, 5)
                gt_vectors = val_images.permute(0, 2, 3, 1).reshape(-1, 5)
                mask_flat = val_mask.reshape(-1)
                valid = mask_flat > 0
                pred_vectors_mask = pred_vectors[valid]
                gt_vectors_mask = gt_vectors[valid]
                cos_sim_mask = torch.nn.functional.cosine_similarity(pred_vectors_mask, gt_vectors_mask, dim=1, eps=1e-8)
                cos_sim_mask = cos_sim_mask.clamp(-1.0, 1.0)
                angles_mask = torch.acos(cos_sim_mask)
                sam_score_mask = angles_mask.mean().item()
                validation_data["validation_metrics"]["SAM_mask"] = sam_score_mask

                # Compute SAM on all 5 channels (global)
                cos_sim_global = torch.nn.functional.cosine_similarity(pred_vectors, gt_vectors, dim=1, eps=1e-8)
                cos_sim_global = cos_sim_global.clamp(-1.0, 1.0)
                angles_global = torch.acos(cos_sim_global)
                sam_score_global = angles_global.mean().item()
                validation_data["validation_metrics"]["SAM_global"] = sam_score_global

                # Per-channel masked MSE logging (mask-only)
                diff = decoded_imgs - val_images
                mask_exp = val_mask.expand_as(diff)
                diff_masked = diff * mask_exp
                mse_num_mask = (diff_masked ** 2).flatten(2).sum(dim=2)
                valid_pixels_mask = mask_exp.flatten(2).sum(dim=2)
                channel_mse_mask = (mse_num_mask / (valid_pixels_mask + 1e-8)).mean(dim=0)
                for idx, val in enumerate(channel_mse_mask):
                    validation_data[f"mse_channel_{idx}_mask"] = float(val)

                # Per-channel global MSE logging
                mse_num_global = (diff ** 2).flatten(2).sum(dim=2)
                valid_pixels_global = torch.ones_like(mask_exp).flatten(2).sum(dim=2)
                channel_mse_global = (mse_num_global / (valid_pixels_global + 1e-8)).mean(dim=0)
                for idx, val in enumerate(channel_mse_global):
                    validation_data[f"mse_channel_{idx}_global"] = float(val)

                # Visualize clamped vs raw output in wandb
                if WANDB_AVAILABLE and wandb.run:
                    # Visualize clamped vs raw output
                    raw_imgs_vis = [Image.fromarray((np.clip(img.cpu().numpy()[0], -1, 1) * 127.5 + 127.5).astype(np.uint8).transpose(1, 2, 0)) for img in decoded_imgs]
                    clamped_imgs_vis = [Image.fromarray((img.cpu().numpy()[0] * 127.5 + 127.5).astype(np.uint8).transpose(1, 2, 0)) for img in decoded_imgs_clamped]
                    wandb_images = []
                    for i, (raw, clamped) in enumerate(zip(raw_imgs_vis, clamped_imgs_vis)):
                        wandb_images.append(wandb.Image(raw, caption=f"raw_{i}"))
                        wandb_images.append(wandb.Image(clamped, caption=f"clamped_{i}"))
                    wandb.log({"val/raw_vs_clamped_outputs": wandb_images}, step=epoch)
            except Exception as e:
                self.log_warning(f"VAE metric computation failed: {e}")
                sam_score_mask = None
                channel_mse_mask = None
                sam_score_global = None
                channel_mse_global = None
        else:
            sam_score_mask = None
            channel_mse_mask = None
            sam_score_global = None
            channel_mse_global = None

        # Write validation summary
        self._write_validation_summary(validation_data)
        self._check_alert_thresholds(validation_data["validation_metrics"], epoch, mode="val")
        # Save JSON backup
        with open(self.json_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
        # Log to wandb
        if WANDB_AVAILABLE and wandb.run:
            wandb_log = {}
            # Add per-channel MSE and metrics
            if channel_mse_mask is not None:
                for idx, val in enumerate(channel_mse_mask):
                    wandb_log[f"val/mse_channel_{idx}_mask"] = float(val)
                for idx, val in enumerate(channel_mse_global):
                    wandb_log[f"val/mse_channel_{idx}_global"] = float(val)
            elif mse_per_channel is not None:
                for idx, val in enumerate(mse_per_channel or []):
                    wandb_log[f"val/mse_channel_{idx}_mask"] = float(val)
                for idx, val in enumerate(mse_per_channel_global or []):
                    wandb_log[f"val/mse_channel_{idx}_global"] = float(val)
            # Add all validation metrics (including range, SSIM, latent, SAM)
            for k, v in validation_data["validation_metrics"].items():
                wandb_log[f"val/{k}"] = v
            if sam_score_mask is not None:
                wandb_log["val/SAM_mask"] = sam_score_mask
                wandb_log["val/SAM_global"] = sam_score_global
            wandb.log(wandb_log, step=epoch)

    def _check_alert_thresholds(self, metrics: Dict[str, float], step_or_epoch: int, mode: str = "step"):
        """Checks key metrics for warning thresholds."""
        prefix = f"[ALERT][{mode.upper()} {step_or_epoch}]"
        if "range/percent_clipped" in metrics:
            val = metrics["range/percent_clipped"]
            if val > 5.0:
                self.log_warning(f"{prefix} CRITICAL: Percent of clipped values >5% ({val:.2f}%)")
            elif val > 1.0:
                self.log_warning(f"{prefix} WARN: Clipping creeping up (>1%) – currently {val:.2f}%")
        if "SAM" in metrics:
            val = metrics["SAM"]
            if val > 0.4:
                self.log_warning(f"{prefix} CRITICAL: SAM > 0.4 indicates angular distortion ({val:.3f})")
            elif val > 0.25:
                self.log_warning(f"{prefix} WARN: SAM > 0.25 may signal degradation ({val:.3f})")
        ssim_vals = [v for k, v in metrics.items() if k.startswith("ssim_channel_")]
        if ssim_vals:
            min_ssim = min(ssim_vals)
            if min_ssim < 0.4:
                self.log_warning(f"{prefix} CRITICAL: One or more SSIM < 0.4 ({min_ssim:.3f})")
            elif min_ssim < 0.6:
                self.log_warning(f"{prefix} WARN: SSIM dropping below 0.6 ({min_ssim:.3f})")

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


def create_dreambooth_logger(output_dir: str, model_name: str = "dreambooth_multispectral", vae=None) -> DreamBoothLogger:
    """
    Factory function to create a DreamBooth training logger.
    
    Args:
        output_dir: Directory to save log files
        model_name: Name for the model/experiment
        vae: The frozen VAE model (optional)
    
    Returns:
        DreamBoothLogger instance
    """
    return DreamBoothLogger(output_dir, model_name, vae=vae)


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
            tags=["dreambooth", "multispectral", "plant-health", "concept-learning", "spectral-imaging"]
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
                           latent_stats: Optional[Dict] = None,
                           memory_usage: Optional[float] = None,
                           validation_images: Optional[List[Image.Image]] = None,
                           validation_prompt: Optional[str] = None,
                           concept_metrics: Optional[Dict] = None,
                           spectral_metrics: Optional[Dict] = None):
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
        latent_stats: VAE latent space statistics
        memory_usage: GPU memory usage
        validation_images: Generated validation images
        validation_prompt: Validation prompt
        concept_metrics: Concept learning metrics
        spectral_metrics: Spectral fidelity metrics
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
        
        # Add latent statistics
        if latent_stats:
            wandb_log_data.update({
                "latent/mean": latent_stats.get('mean', 0),
                "latent/std": latent_stats.get('std', 0),
                "latent/min": latent_stats.get('min', 0),
                "latent/max": latent_stats.get('max', 0),
            })
        
        # Add concept metrics
        if concept_metrics:
            wandb_log_data.update({
                "concept/similarity": concept_metrics.get('similarity', 0),
                "concept/preservation": concept_metrics.get('preservation', 0),
                "concept/quality": concept_metrics.get('quality', 0),
            })
        
        # Add spectral metrics
        if spectral_metrics:
            wandb_log_data.update({
                "spectral/fidelity": spectral_metrics.get('fidelity', 0),
                "spectral/latent_quality": spectral_metrics.get('latent_quality', 0),
                "spectral/conversion_quality": spectral_metrics.get('conversion_quality', 0),
            })
        
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


def create_spectral_visualization(original_image: torch.Tensor, 
                                generated_image: Image.Image,
                                band_names: List[str] = None) -> plt.Figure:
    """
    Create a spectral visualization comparing original multispectral data with generated image.
    
    Args:
        original_image: Original 5-channel multispectral tensor (B, 5, H, W)
        generated_image: Generated RGB image from PIL
        band_names: Names of spectral bands
    
    Returns:
        matplotlib Figure with spectral comparison
    """
    if band_names is None:
        band_names = ["Band 9 (474nm)", "Band 18 (539nm)", "Band 32 (651nm)", 
                     "Band 42 (731nm)", "Band 55 (851nm)"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Spectral Comparison: Original vs Generated", fontsize=16)
    
    # Plot original multispectral bands
    for i in range(5):
        row = i // 3
        col = i % 3
        band_data = original_image[0, i].detach().cpu().numpy()
        
        # Normalize to [0, 1] for visualization
        band_data = (band_data + 1.0) / 2.0
        band_data = np.clip(band_data, 0, 1)
        
        im = axes[row, col].imshow(band_data, cmap='viridis')
        axes[row, col].set_title(band_names[i])
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    # Plot generated RGB image
    axes[1, 2].imshow(generated_image)
    axes[1, 2].set_title("Generated RGB")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig


def analyze_concept_learning(text_encoder_one, text_encoder_two, text_encoder_three,
                           tokenizer_one, tokenizer_two, tokenizer_three,
                           instance_prompt: str, class_prompt: str,
                           device: torch.device) -> Dict[str, float]:
    """
    Analyze concept learning by comparing instance and class embeddings.
    
    Args:
        text_encoder_one/two/three: Text encoders
        tokenizer_one/two/three: Tokenizers
        instance_prompt: Instance prompt (e.g., "sks leaf")
        class_prompt: Class prompt (e.g., "leaf")
        device: Device to use
    
    Returns:
        Dictionary with concept learning metrics
    """
    try:
        # Encode instance prompt
        instance_embeddings = []
        for encoder, tokenizer in [(text_encoder_one, tokenizer_one), 
                                  (text_encoder_two, tokenizer_two),
                                  (text_encoder_three, tokenizer_three)]:
            with torch.no_grad():
                inputs = tokenizer(instance_prompt, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = encoder(**inputs)
                instance_embeddings.append(outputs.last_hidden_state.mean(dim=1))
        
        # Encode class prompt
        class_embeddings = []
        for encoder, tokenizer in [(text_encoder_one, tokenizer_one), 
                                  (text_encoder_two, tokenizer_two),
                                  (text_encoder_three, tokenizer_three)]:
            with torch.no_grad():
                inputs = tokenizer(class_prompt, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = encoder(**inputs)
                class_embeddings.append(outputs.last_hidden_state.mean(dim=1))
        
        # Calculate similarities
        similarities = []
        for inst_emb, class_emb in zip(instance_embeddings, class_embeddings):
            sim = torch.cosine_similarity(inst_emb, class_emb, dim=1)
            similarities.append(sim.item())
        
        avg_similarity = np.mean(similarities)
        
        return {
            "similarity": avg_similarity,
            "similarities_per_encoder": similarities,
            "concept_quality": min(avg_similarity, 1.0),  # Higher similarity = better concept learning
            "preservation_score": 1.0 - abs(avg_similarity - 0.5)  # Optimal around 0.5
        }
        
    except Exception as e:
        print(f"Warning: Failed to analyze concept learning: {e}")
        return {
            "similarity": 0.0,
            "similarities_per_encoder": [0.0, 0.0, 0.0],
            "concept_quality": 0.0,
            "preservation_score": 0.0
        }


def compute_latent_statistics(latent_tensor: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics of VAE latent space.
    
    Args:
        latent_tensor: Latent tensor from VAE (B, C, H, W)
    
    Returns:
        Dictionary with latent statistics
    """
    with torch.no_grad():
        return {
            "mean": latent_tensor.mean().item(),
            "std": latent_tensor.std().item(),
            "min": latent_tensor.min().item(),
            "max": latent_tensor.max().item(),
            "spatial_mean": latent_tensor.mean(dim=(2, 3)).mean().item(),
            "channel_std": latent_tensor.std(dim=(2, 3)).mean().item(),
        }


def compute_spectral_fidelity_metrics(original_image: torch.Tensor,
                                    generated_image: Image.Image) -> Dict[str, float]:
    """
    Compute spectral fidelity metrics between original and generated images.
    
    Args:
        original_image: Original 5-channel multispectral tensor
        generated_image: Generated RGB image
    
    Returns:
        Dictionary with spectral fidelity metrics
    """
    try:
        # Convert generated image to tensor
        generated_tensor = torch.tensor(np.array(generated_image)).float() / 255.0
        generated_tensor = generated_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Resize original to match generated
        if original_image.shape[2:] != generated_tensor.shape[2:]:
            original_image = torch.nn.functional.interpolate(
                original_image, size=generated_tensor.shape[2:], mode='bilinear'
            )
        
        # Compute basic statistics
        orig_mean = original_image.mean().item()
        orig_std = original_image.std().item()
        gen_mean = generated_tensor.mean().item()
        gen_std = generated_tensor.std().item()
        
        # Compute structural similarity (using first 3 bands vs RGB)
        orig_rgb = original_image[:, :3]  # Use first 3 bands as RGB approximation
        ssim_score = 0.0
        
        try:
            from skimage.metrics import structural_similarity as ssim
            for i in range(3):
                orig_band = orig_rgb[0, i].detach().cpu().numpy()
                gen_band = generated_tensor[0, i].detach().cpu().numpy()
                ssim_score += ssim(orig_band, gen_band, data_range=1.0)
            ssim_score /= 3.0
        except ImportError:
            ssim_score = 0.0
        
        return {
            "fidelity": ssim_score,
            "mean_difference": abs(orig_mean - gen_mean),
            "std_difference": abs(orig_std - gen_std),
            "latent_quality": 1.0 - abs(orig_mean - gen_mean),  # Higher if means are similar
            "conversion_quality": ssim_score,  # Quality of multispectral to RGB conversion
        }
        
    except Exception as e:
        print(f"Warning: Failed to compute spectral fidelity: {e}")
        return {
            "fidelity": 0.0,
            "mean_difference": 1.0,
            "std_difference": 1.0,
            "latent_quality": 0.0,
            "conversion_quality": 0.0,
        }


def compute_band_importance_analysis(vae_model) -> Dict[str, float]:
    """
    Compute band importance analysis from VAE model.
    
    Args:
        vae_model: Multispectral VAE model with attention mechanisms
    
    Returns:
        Dictionary with band importance scores
    """
    try:
        if hasattr(vae_model, 'input_adapter') and hasattr(vae_model.input_adapter, 'attention'):
            return vae_model.input_adapter.attention.get_band_importance()
        elif hasattr(vae_model, 'output_adapter') and hasattr(vae_model.output_adapter, 'attention'):
            return vae_model.output_adapter.attention.get_band_importance()
        else:
            return {}
    except Exception as e:
        print(f"Warning: Failed to compute band importance: {e}")
        return {}


def compute_band_correlation_analysis(original_image: torch.Tensor,
                                    generated_image: Image.Image) -> Dict[str, float]:
    """
    Compute correlation between original and generated spectral bands.
    
    Args:
        original_image: Original 5-channel multispectral tensor
        generated_image: Generated RGB image
    
    Returns:
        Dictionary with band correlation scores
    """
    try:
        # Convert generated image to tensor
        generated_tensor = torch.tensor(np.array(generated_image)).float() / 255.0
        generated_tensor = generated_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Resize original to match generated
        if original_image.shape[2:] != generated_tensor.shape[2:]:
            original_image = torch.nn.functional.interpolate(
                original_image, size=generated_tensor.shape[2:], mode='bilinear'
            )
        
        correlations = {}
        
        # Compute correlation for first 3 bands
        for i in range(min(3, original_image.shape[1])):
            orig_band = original_image[0, i].flatten().detach().cpu().numpy()
            gen_band = generated_tensor[0, i].flatten().detach().cpu().numpy()
            
            # Compute Pearson correlation
            correlation = np.corrcoef(orig_band, gen_band)[0, 1]
            if not np.isnan(correlation):
                correlations[f"band_{i+1}"] = correlation
        
        return correlations
        
    except Exception as e:
        print(f"Warning: Failed to compute band correlation: {e}")
        return {}


def compute_spectral_signature_preservation(original_image: torch.Tensor,
                                          generated_image: Image.Image) -> float:
    """
    Compute spectral signature preservation using SAM-like approach.
    
    Args:
        original_image: Original 5-channel multispectral tensor
        generated_image: Generated RGB image
    
    Returns:
        Spectral signature preservation score (0-1)
    """
    try:
        # Convert generated image to tensor
        generated_tensor = torch.tensor(np.array(generated_image)).float() / 255.0
        generated_tensor = generated_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Resize original to match generated
        if original_image.shape[2:] != generated_tensor.shape[2:]:
            original_image = torch.nn.functional.interpolate(
                original_image, size=generated_tensor.shape[2:], mode='bilinear'
            )
        
        # Use first 3 bands for comparison
        orig_rgb = original_image[:, :3]
        
        # Compute spectral angle mapper (SAM) for each pixel
        sam_scores = []
        
        for h in range(orig_rgb.shape[2]):
            for w in range(orig_rgb.shape[3]):
                orig_spectrum = orig_rgb[0, :, h, w].detach().cpu().numpy()
                gen_spectrum = generated_tensor[0, :, h, w].detach().cpu().numpy()
                
                # Compute spectral angle
                dot_product = np.dot(orig_spectrum, gen_spectrum)
                orig_norm = np.linalg.norm(orig_spectrum)
                gen_norm = np.linalg.norm(gen_spectrum)
                
                if orig_norm > 0 and gen_norm > 0:
                    cos_angle = dot_product / (orig_norm * gen_norm)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    sam_scores.append(angle)
        
        if sam_scores:
            # Convert to similarity score (0-1, higher is better)
            mean_angle = np.mean(sam_scores)
            similarity = 1.0 - (mean_angle / np.pi)  # Normalize to [0, 1]
            return float(similarity)
        else:
            return 0.0
        
    except Exception as e:
        print(f"Warning: Failed to compute spectral signature preservation: {e}")
        return 0.0


def compute_latent_clustering_analysis(latent_tensor: torch.Tensor) -> Dict[str, float]:
    """
    Compute latent space clustering analysis.
    
    Args:
        latent_tensor: VAE latent tensor (B, C, H, W)
    
    Returns:
        Dictionary with clustering metrics
    """
    try:
        with torch.no_grad():
            # Flatten spatial dimensions
            latent_flat = latent_tensor.view(latent_tensor.shape[0], -1)  # (B, C*H*W)
            
            # Compute basic clustering metrics
            mean_dist = torch.pdist(latent_flat).mean().item()
            std_dist = torch.pdist(latent_flat).std().item()
            
            # Compute separation score (higher is better)
            separation = 1.0 / (1.0 + mean_dist)
            
            # Compute clustering score based on variance
            clustering_score = 1.0 / (1.0 + std_dist)
            
            return {
                "clustering_score": clustering_score,
                "separation": separation,
                "mean_distance": mean_dist,
                "std_distance": std_dist,
            }
        
    except Exception as e:
        print(f"Warning: Failed to compute latent clustering: {e}")
        return {
            "clustering_score": 0.0,
            "separation": 0.0,
            "mean_distance": 1.0,
            "std_distance": 1.0,
        }


def compute_concept_latent_mapping(text_embeddings: torch.Tensor,
                                 latent_tensor: torch.Tensor) -> Dict[str, float]:
    """
    Compute mapping between concept embeddings and latent representations.
    
    Args:
        text_embeddings: Text encoder embeddings
        latent_tensor: VAE latent tensor
    
    Returns:
        Dictionary with concept-latent mapping metrics
    """
    try:
        with torch.no_grad():
            # Flatten latent tensor
            latent_flat = latent_tensor.view(latent_tensor.shape[0], -1)  # (B, C*H*W)
            
            # Compute correlation between text and latent
            text_mean = text_embeddings.mean(dim=1)  # (B, D)
            
            # Compute correlation for each batch
            correlations = []
            for i in range(text_mean.shape[0]):
                if i < latent_flat.shape[0]:
                    text_vec = text_mean[i].cpu().numpy()
                    latent_vec = latent_flat[i].cpu().numpy()
                    
                    # Compute correlation
                    correlation = np.corrcoef(text_vec, latent_vec)[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(correlation)
            
            if correlations:
                mean_correlation = np.mean(correlations)
                separation = 1.0 - abs(mean_correlation)  # Higher separation if correlation is low
                
                return {
                    "correlation": mean_correlation,
                    "separation": separation,
                    "correlation_std": np.std(correlations),
                }
            else:
                return {
                    "correlation": 0.0,
                    "separation": 0.0,
                    "correlation_std": 0.0,
                }
        
    except Exception as e:
        print(f"Warning: Failed to compute concept-latent mapping: {e}")
        return {
            "correlation": 0.0,
            "separation": 0.0,
            "correlation_std": 0.0,
        }


def create_advanced_spectral_visualization(original_image: torch.Tensor,
                                         generated_image: Image.Image,
                                         band_names: List[str] = None,
                                         concept_embeddings: Optional[torch.Tensor] = None,
                                         latent_tensor: Optional[torch.Tensor] = None) -> plt.Figure:
    """
    Create advanced spectral visualization with concept and latent analysis.
    
    Args:
        original_image: Original 5-channel multispectral tensor
        generated_image: Generated RGB image
        band_names: Names of spectral bands
        concept_embeddings: Text encoder embeddings
        latent_tensor: VAE latent tensor
    
    Returns:
        matplotlib Figure with comprehensive analysis
    """
    if band_names is None:
        band_names = ["Band 9 (474nm)", "Band 18 (539nm)", "Band 32 (651nm)", 
                     "Band 42 (731nm)", "Band 55 (851nm)"]
    
    # Create subplot grid
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # Original multispectral bands
    for i in range(5):
        row = i // 3
        col = i % 3
        band_data = original_image[0, i].detach().cpu().numpy()
        band_data = (band_data + 1.0) / 2.0
        band_data = np.clip(band_data, 0, 1)
        
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(band_data, cmap='viridis')
        ax.set_title(band_names[i], fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Generated RGB image
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(generated_image)
    ax.set_title("Generated RGB", fontsize=10)
    ax.axis('off')
    
    # Spectral signature comparison
    ax = fig.add_subplot(gs[2, :2])
    orig_rgb = original_image[:, :3]  # First 3 bands
    orig_mean = orig_rgb.mean(dim=(2, 3))[0].detach().cpu().numpy()
    
    generated_tensor = torch.tensor(np.array(generated_image)).float() / 255.0
    generated_tensor = generated_tensor.permute(2, 0, 1).unsqueeze(0)
    gen_mean = generated_tensor.mean(dim=(2, 3))[0].detach().cpu().numpy()
    
    x = np.arange(3)
    width = 0.35
    
    ax.bar(x - width/2, orig_mean, width, label='Original', alpha=0.7)
    ax.bar(x + width/2, gen_mean, width, label='Generated', alpha=0.7)
    ax.set_xlabel('Band')
    ax.set_ylabel('Mean Intensity')
    ax.set_title('Spectral Signature Comparison')
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(['R', 'G', 'B'])
    
    # Latent space visualization (if available)
    if latent_tensor is not None:
        ax = fig.add_subplot(gs[2, 2:])
        latent_flat = latent_tensor[0].detach().cpu().numpy()
        latent_mean = latent_flat.mean(axis=(1, 2))
        
        ax.plot(latent_mean, 'b-', alpha=0.7, label='Latent Mean')
        ax.fill_between(range(len(latent_mean)), 
                       latent_mean - latent_flat.std(axis=(1, 2)),
                       latent_mean + latent_flat.std(axis=(1, 2)),
                       alpha=0.3)
        ax.set_xlabel('Latent Channel')
        ax.set_ylabel('Activation')
        ax.set_title('Latent Space Analysis')
        ax.legend()
    
    # Concept embedding analysis (if available)
    if concept_embeddings is not None:
        ax = fig.add_subplot(gs[3, :])
        concept_mean = concept_embeddings.mean(dim=1)[0].detach().cpu().numpy()
        
        ax.plot(concept_mean, 'r-', alpha=0.7, label='Concept Embedding')
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('Activation')
        ax.set_title('Concept Embedding Analysis')
        ax.legend()
    
    plt.suptitle("Advanced Spectral Analysis: Original vs Generated", fontsize=16)
    return fig 