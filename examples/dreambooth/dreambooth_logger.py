"""
DreamBooth Multispectral Training Logger

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
import wandb
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


class DreamBoothLogger:
    """
    Comprehensive training logger for DreamBooth multispectral training.
    
    Captures and logs:
    - Training progress metrics
    - Concept learning indicators
    - Spectral fidelity metrics
    - Validation performance
    - Generated image quality
    - System performance
    
    SCIENTIFIC CONTRIBUTION:
    ------------------------
    This logger addresses the unique challenges of multispectral DreamBooth training by providing:
    
    1. CONCEPT LEARNING ANALYSIS:
       - Monitors "sks" concept learning progress
       - Tracks prior preservation loss effectiveness
       - Analyzes text encoder adaptation to multispectral concepts
       - Provides interpretable patterns of concept-spectral mapping
    
    2. SPECTRAL FIDELITY MONITORING:
       - Quantifies spectral signature preservation during training
       - Monitors VAE latent space quality
       - Tracks multispectral-to-RGB conversion effectiveness
       - Enables detection of spectral information loss
    
    3. TRAINING STABILITY ASSESSMENT:
       - Tracks gradient clipping events for optimization stability
       - Monitors mixed precision training performance
       - Provides comprehensive loss decomposition
       - Enables early detection of training divergence
    
    IMPLEMENTATION DESIGN:
    ----------------------
    The logger employs a hierarchical data structure that separates:
    - Step-level metrics (training progress, loss values)
    - Epoch-level metrics (validation performance, concept quality)
    - Spectral-specific analysis (fidelity, latent quality)
    - System-level information (timing, resource usage)
    
    This design enables both real-time monitoring and comprehensive post-training analysis.
    """
    
    def __init__(self, output_dir: str, model_name: str = "dreambooth_multispectral"):
        """
        Initialize the DreamBooth training logger.
        
        Args:
            output_dir: Directory to save log files
            model_name: Name for the model/experiment
        """
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.log_dir = self.output_dir / "dreambooth_logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{model_name}_training_log_{timestamp}.txt"
        self.json_file = self.log_dir / f"{model_name}_training_log_{timestamp}.json"
        
        # Initialize log data
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
            "concept_analysis": [],
            "system_performance": [],
            "best_metrics": {},
            "final_summary": {}
        }
        
        # Track best metrics
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_step = 0
        
        # Write header
        self._write_header()
    
    def _write_header(self):
        """Write header information to the log file."""
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
        
        config_str = f"""
TRAINING CONFIGURATION:
{'-'*40}
"""
        for key, value in config.items():
            config_str += f"{key}: {value}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(config_str + "\n")
    
    def log_step(self, 
                 step: int,
                 epoch: int,
                 loss: float,
                 learning_rate: float,
                 grad_norm: Optional[float] = None,
                 gradient_clipping: Optional[bool] = None,
                 prior_loss: Optional[float] = None,
                 instance_loss: Optional[float] = None,
                 latent_stats: Optional[Dict] = None,
                 memory_usage: Optional[float] = None,
                 training_speed: Optional[float] = None,
                 gpu_utilization: Optional[float] = None,
                 cpu_utilization: Optional[float] = None,
                 concept_similarity: Optional[float] = None,
                 prior_preservation_score: Optional[float] = None,
                 spectral_fidelity: Optional[float] = None,
                 conversion_quality: Optional[float] = None):
        """
        Log step-level metrics.
        
        This method captures detailed training progress including concept learning
        indicators that are critical for DreamBooth training analysis.
        
        DREAMBOOTH-SPECIFIC METRICS:
        ----------------------------
        
        1. CONCEPT LEARNING TRACKING:
           - loss: Total training loss
           - prior_loss: Prior preservation loss (if enabled)
           - instance_loss: Instance-specific loss
           - These enable assessment of concept learning vs. preservation balance
        
        2. TRAINING STABILITY:
           - grad_norm: Gradient magnitude for stability assessment
           - gradient_clipping: Indicates optimization instability
           - learning_rate: Current learning rate for scheduling analysis
        
        3. SPECTRAL FIDELITY:
           - latent_stats: VAE latent space statistics
           - Memory usage for computational efficiency
        
        Args:
            step: Current training step
            epoch: Current epoch
            loss: Total training loss
            learning_rate: Current learning rate
            grad_norm: Gradient norm
            gradient_clipping: Whether gradient clipping occurred
            prior_loss: Prior preservation loss (if enabled)
            instance_loss: Instance-specific loss
            latent_stats: VAE latent space statistics
            memory_usage: GPU memory usage in GB
        """
        step_data = {
            "step": step,
            "epoch": epoch,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "loss": loss,
            "learning_rate": learning_rate,
            "grad_norm": grad_norm,
            "gradient_clipping": gradient_clipping,
            "prior_loss": prior_loss,
            "instance_loss": instance_loss,
            "latent_stats": latent_stats,
            "memory_usage": memory_usage,
            "training_speed": training_speed,
            "gpu_utilization": gpu_utilization,
            "cpu_utilization": cpu_utilization,
            "concept_similarity": concept_similarity,
            "prior_preservation_score": prior_preservation_score,
            "spectral_fidelity": spectral_fidelity,
            "conversion_quality": conversion_quality
        }
        
        self.log_data["steps"].append(step_data)
        
        # Write to text file (only every 100 steps to avoid spam)
        if step % 100 == 0:
            self._write_step_summary(step_data)
        
        # Log to wandb every 10 steps for real-time monitoring
        if step % 10 == 0 and wandb.run:
            self._log_step_to_wandb(step_data)
        
        # Save JSON backup
        with open(self.json_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
    
    def _write_step_summary(self, step_data: Dict[str, Any]):
        """Write step summary to text file."""
        step = step_data["step"]
        epoch = step_data["epoch"]
        loss = step_data["loss"]
        lr = step_data["learning_rate"]
        grad_norm = step_data.get("grad_norm", 0)
        gradient_clipping = step_data.get("gradient_clipping", False)
        prior_loss = step_data.get("prior_loss", None)
        instance_loss = step_data.get("instance_loss", None)
        memory_usage = step_data.get("memory_usage", 0)
        training_speed = step_data.get("training_speed", 0)
        gpu_util = step_data.get("gpu_utilization", 0)
        cpu_util = step_data.get("cpu_utilization", 0)
        concept_sim = step_data.get("concept_similarity", 0)
        prior_pres = step_data.get("prior_preservation_score", 0)
        spectral_fid = step_data.get("spectral_fidelity", 0)
        conv_quality = step_data.get("conversion_quality", 0)
        
        # Format loss components
        loss_str = f"Loss: {loss:.6f}"
        if prior_loss is not None:
            loss_str += f" | Prior: {prior_loss:.6f}"
        if instance_loss is not None:
            loss_str += f" | Instance: {instance_loss:.6f}"
        
        # Format gradient clipping indicator
        clip_str = " [CLIP]" if gradient_clipping else ""
        
        # Format latent stats if available
        latent_str = ""
        if step_data.get("latent_stats"):
            stats = step_data["latent_stats"]
            latent_str = f" | Latent: μ={stats.get('mean', 0):.3f}, σ={stats.get('std', 0):.3f}"
        
        # Format performance metrics
        perf_str = ""
        if training_speed > 0:
            perf_str += f" | Speed: {training_speed:.1f} steps/s"
        if gpu_util > 0:
            perf_str += f" | GPU: {gpu_util:.1f}%"
        if cpu_util > 0:
            perf_str += f" | CPU: {cpu_util:.1f}%"
        
        # Format concept and spectral metrics
        concept_str = ""
        if concept_sim > 0:
            concept_str += f" | Concept: {concept_sim:.3f}"
        if prior_pres > 0:
            concept_str += f" | Prior: {prior_pres:.3f}"
        if spectral_fid > 0:
            concept_str += f" | Spectral: {spectral_fid:.3f}"
        if conv_quality > 0:
            concept_str += f" | Conv: {conv_quality:.3f}"
        
        # Main step line
        step_line = f"Step {step:6d} (Epoch {epoch:3d}) | {loss_str} | LR: {lr:.2e} | Grad: {grad_norm:.3f}{clip_str} | Mem: {memory_usage:.1f}GB{latent_str}{perf_str}{concept_str}"
        
        with open(self.log_file, 'a') as f:
            f.write(step_line + "\n")
    
    def _log_step_to_wandb(self, step_data: Dict[str, Any]):
        """Log step data to wandb for real-time monitoring."""
        if not wandb.run:
            return
        
        try:
            wandb_log_data = {
                "step": step_data["step"],
                "epoch": step_data["epoch"],
                "loss": step_data["loss"],
                "learning_rate": step_data["learning_rate"],
            }
            
            # Add optional metrics
            if step_data.get("grad_norm"):
                wandb_log_data["gradient_norm"] = step_data["grad_norm"]
            if step_data.get("prior_loss"):
                wandb_log_data["prior_loss"] = step_data["prior_loss"]
            if step_data.get("instance_loss"):
                wandb_log_data["instance_loss"] = step_data["instance_loss"]
            if step_data.get("memory_usage"):
                wandb_log_data["memory_usage_gb"] = step_data["memory_usage"]
            if step_data.get("training_speed"):
                wandb_log_data["training_speed"] = step_data["training_speed"]
            if step_data.get("gpu_utilization"):
                wandb_log_data["gpu_utilization"] = step_data["gpu_utilization"]
            if step_data.get("cpu_utilization"):
                wandb_log_data["cpu_utilization"] = step_data["cpu_utilization"]
            if step_data.get("concept_similarity"):
                wandb_log_data["concept_similarity"] = step_data["concept_similarity"]
            if step_data.get("prior_preservation_score"):
                wandb_log_data["prior_preservation_score"] = step_data["prior_preservation_score"]
            if step_data.get("spectral_fidelity"):
                wandb_log_data["spectral_fidelity"] = step_data["spectral_fidelity"]
            if step_data.get("conversion_quality"):
                wandb_log_data["conversion_quality"] = step_data["conversion_quality"]
            
            # Add latent statistics
            if step_data.get("latent_stats"):
                stats = step_data["latent_stats"]
                wandb_log_data.update({
                    "latent/mean": stats.get("mean", 0),
                    "latent/std": stats.get("std", 0),
                    "latent/min": stats.get("min", 0),
                    "latent/max": stats.get("max", 0),
                    "latent/spatial_mean": stats.get("spatial_mean", 0),
                    "latent/channel_std": stats.get("channel_std", 0),
                })
            
            wandb.log(wandb_log_data)
            
        except Exception as e:
            print(f"Warning: Failed to log step to wandb: {e}")
    
    def _log_epoch_to_wandb(self, epoch_data: Dict[str, Any]):
        """Log epoch data to wandb for comprehensive monitoring."""
        if not wandb.run:
            return
        
        try:
            wandb_log_data = {
                "epoch": epoch_data["epoch"],
                "avg_loss": epoch_data["avg_loss"],
                "avg_learning_rate": epoch_data["avg_learning_rate"],
                "total_steps": epoch_data["total_steps"],
                "epoch_time": epoch_data["epoch_time"],
            }
            
            # Add validation metrics
            if epoch_data.get("validation_metrics"):
                val_metrics = epoch_data["validation_metrics"]
                wandb_log_data.update({
                    "val_loss": val_metrics.get("val_loss", 0),
                    "val_concept_similarity": val_metrics.get("concept_similarity", 0),
                    "val_image_quality": val_metrics.get("image_quality", 0),
                })
            
            # Add concept quality metrics
            if epoch_data.get("concept_quality"):
                concept = epoch_data["concept_quality"]
                wandb_log_data.update({
                    "concept_quality": concept.get("quality_score", 0),
                    "concept_prior_preservation": concept.get("prior_preservation", 0),
                    "concept_similarity": concept.get("similarity", 0),
                })
            
            # Add spectral fidelity metrics
            if epoch_data.get("spectral_fidelity"):
                spectral = epoch_data["spectral_fidelity"]
                wandb_log_data.update({
                    "spectral_fidelity": spectral.get("fidelity", 0),
                    "spectral_latent_quality": spectral.get("latent_quality", 0),
                    "spectral_conversion_quality": spectral.get("conversion_quality", 0),
                })
            
            # Add band importance
            if epoch_data.get("band_importance"):
                band_imp = epoch_data["band_importance"]
                if isinstance(band_imp, dict):
                    for band, importance in band_imp.items():
                        wandb_log_data[f"band_importance_{band}"] = importance
            
            # Add band correlation
            if epoch_data.get("band_correlation"):
                band_corr = epoch_data["band_correlation"]
                if isinstance(band_corr, dict):
                    for band_pair, correlation in band_corr.items():
                        wandb_log_data[f"band_correlation_{band_pair}"] = correlation
            
            # Add spectral signature preservation
            if epoch_data.get("spectral_signature_preservation"):
                wandb_log_data["spectral_signature_preservation"] = epoch_data["spectral_signature_preservation"]
            
            # Add latent clustering metrics
            if epoch_data.get("latent_clustering"):
                latent_cluster = epoch_data["latent_clustering"]
                wandb_log_data.update({
                    "latent_clustering_score": latent_cluster.get("clustering_score", 0),
                    "latent_separation": latent_cluster.get("separation", 0),
                })
            
            # Add concept-latent mapping metrics
            if epoch_data.get("concept_latent_mapping"):
                concept_map = epoch_data["concept_latent_mapping"]
                wandb_log_data.update({
                    "concept_latent_correlation": concept_map.get("correlation", 0),
                    "concept_latent_separation": concept_map.get("separation", 0),
                })
            
            wandb.log(wandb_log_data)
            
        except Exception as e:
            print(f"Warning: Failed to log epoch to wandb: {e}")
    
    def log_epoch(self, 
                  epoch: int,
                  avg_loss: float,
                  avg_learning_rate: float,
                  total_steps: int,
                  epoch_time: float,
                  validation_metrics: Optional[Dict] = None,
                  concept_quality: Optional[Dict] = None,
                  spectral_fidelity: Optional[Dict] = None,
                  band_importance: Optional[Dict] = None,
                  band_correlation: Optional[Dict] = None,
                  spectral_signature_preservation: Optional[float] = None,
                  latent_clustering: Optional[Dict] = None,
                  concept_latent_mapping: Optional[Dict] = None):
        """
        Log epoch-level metrics.
        
        Args:
            epoch: Current epoch number
            avg_loss: Average loss for the epoch
            avg_learning_rate: Average learning rate for the epoch
            total_steps: Total steps in the epoch
            epoch_time: Time taken for the epoch
            validation_metrics: Validation performance metrics
            concept_quality: Concept learning quality indicators
            spectral_fidelity: Spectral fidelity metrics
        """
        epoch_data = {
            "epoch": epoch,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "avg_loss": avg_loss,
            "avg_learning_rate": avg_learning_rate,
            "total_steps": total_steps,
            "epoch_time": epoch_time,
            "validation_metrics": validation_metrics,
            "concept_quality": concept_quality,
            "spectral_fidelity": spectral_fidelity,
            "band_importance": band_importance,
            "band_correlation": band_correlation,
            "spectral_signature_preservation": spectral_signature_preservation,
            "latent_clustering": latent_clustering,
            "concept_latent_mapping": concept_latent_mapping
        }
        
        self.log_data["epochs"].append(epoch_data)
        
        # Check for best model
        if validation_metrics and validation_metrics.get('val_loss', float('inf')) < self.best_val_loss:
            self.best_val_loss = validation_metrics['val_loss']
            self.best_epoch = epoch
            self.log_data["best_metrics"] = {
                "best_epoch": epoch,
                "best_val_loss": self.best_val_loss,
                "train_loss_at_best": avg_loss
            }
        
        # Write to text file
        self._write_epoch_summary(epoch_data)
        
        # Save JSON backup
        with open(self.json_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
    
    def _write_epoch_summary(self, epoch_data: Dict[str, Any]):
        """Write epoch summary to text file."""
        epoch = epoch_data["epoch"]
        avg_loss = epoch_data["avg_loss"]
        avg_lr = epoch_data["avg_learning_rate"]
        total_steps = epoch_data["total_steps"]
        epoch_time = epoch_data["epoch_time"]
        
        # Format validation metrics
        val_str = ""
        if epoch_data.get("validation_metrics"):
            val_metrics = epoch_data["validation_metrics"]
            val_str = f" | Val Loss: {val_metrics.get('val_loss', 0):.6f}"
            if 'concept_similarity' in val_metrics:
                val_str += f" | Concept Sim: {val_metrics['concept_similarity']:.3f}"
            if 'image_quality' in val_metrics:
                val_str += f" | Quality: {val_metrics['image_quality']:.3f}"
        
        # Format concept quality
        concept_str = ""
        if epoch_data.get("concept_quality"):
            concept = epoch_data["concept_quality"]
            concept_str = f" | Concept Quality: {concept.get('quality_score', 0):.3f}"
            if 'prior_preservation' in concept:
                concept_str += f" | Prior Pres: {concept['prior_preservation']:.3f}"
            if 'similarity' in concept:
                concept_str += f" | Similarity: {concept['similarity']:.3f}"
        
        # Format spectral fidelity
        spectral_str = ""
        if epoch_data.get("spectral_fidelity"):
            spectral = epoch_data["spectral_fidelity"]
            spectral_str = f" | Spectral Fidelity: {spectral.get('fidelity', 0):.3f}"
            if 'latent_quality' in spectral:
                spectral_str += f" | Latent: {spectral['latent_quality']:.3f}"
            if 'conversion_quality' in spectral:
                spectral_str += f" | Conv: {spectral['conversion_quality']:.3f}"
        
        # Format band importance
        band_str = ""
        if epoch_data.get("band_importance"):
            band_imp = epoch_data["band_importance"]
            if isinstance(band_imp, dict) and len(band_imp) > 0:
                # Show top 3 bands
                sorted_bands = sorted(band_imp.items(), key=lambda x: x[1], reverse=True)[:3]
                band_str = f" | Top Bands: {', '.join([f'{k}({v:.2f})' for k, v in sorted_bands])}"
        
        # Format spectral signature preservation
        sig_str = ""
        if epoch_data.get("spectral_signature_preservation"):
            sig_pres = epoch_data["spectral_signature_preservation"]
            sig_str = f" | Sig Pres: {sig_pres:.3f}"
        
        # Main epoch line
        epoch_line = f"Epoch {epoch:3d} | Avg Loss: {avg_loss:.6f} | Steps: {total_steps} | Time: {epoch_time:.1f}s | LR: {avg_lr:.2e}{val_str}{concept_str}{spectral_str}{band_str}{sig_str}"
        
        # Add best model indicator
        if epoch == self.best_epoch:
            epoch_line += " [BEST]"
        
        with open(self.log_file, 'a') as f:
            f.write(epoch_line + "\n")
        
        # Log to wandb
        if wandb.run:
            self._log_epoch_to_wandb(epoch_data)
    
    def log_validation(self, 
                      epoch: int,
                      images: List[Image.Image],
                      prompt: str,
                      validation_metrics: Dict[str, Any],
                      spectral_analysis: Optional[Dict] = None):
        """
        Log validation results including generated images and metrics.
        
        Args:
            epoch: Current epoch
            images: Generated validation images
            prompt: Validation prompt used
            validation_metrics: Validation performance metrics
            spectral_analysis: Spectral analysis of generated images
        """
        validation_data = {
            "epoch": epoch,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompt,
            "num_images": len(images),
            "validation_metrics": validation_metrics,
            "spectral_analysis": spectral_analysis
        }
        
        self.log_data["validations"].append(validation_data)
        
        # Save validation images
        validation_dir = self.log_dir / "validation_images"
        validation_dir.mkdir(exist_ok=True)
        
        for i, image in enumerate(images):
            image_path = validation_dir / f"epoch_{epoch:03d}_image_{i:02d}.png"
            image.save(image_path)
        
        # Write validation summary
        self._write_validation_summary(validation_data)
        
        # Save JSON backup
        with open(self.json_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
    
    def _write_validation_summary(self, validation_data: Dict[str, Any]):
        """Write validation summary to text file."""
        epoch = validation_data["epoch"]
        prompt = validation_data["prompt"]
        num_images = validation_data["num_images"]
        metrics = validation_data["validation_metrics"]
        
        val_line = f"Validation Epoch {epoch:3d} | Prompt: '{prompt}' | Images: {num_images}"
        
        if metrics:
            val_line += f" | Loss: {metrics.get('val_loss', 0):.6f}"
            if 'concept_similarity' in metrics:
                val_line += f" | Concept Sim: {metrics['concept_similarity']:.3f}"
            if 'image_quality' in metrics:
                val_line += f" | Quality: {metrics['image_quality']:.3f}"
        
        with open(self.log_file, 'a') as f:
            f.write(val_line + "\n")
    
    def log_concept_analysis(self, 
                           epoch: int,
                           concept_embeddings: Optional[torch.Tensor] = None,
                           prior_embeddings: Optional[torch.Tensor] = None,
                           concept_similarity: Optional[float] = None,
                           prior_preservation_score: Optional[float] = None,
                           text_encoder_adaptation: Optional[Dict] = None):
        """
        Log concept learning analysis.
        
        Args:
            epoch: Current epoch
            concept_embeddings: Text embeddings for the concept
            prior_embeddings: Text embeddings for the prior
            concept_similarity: Similarity between concept and prior
            prior_preservation_score: Prior preservation effectiveness
            text_encoder_adaptation: Text encoder adaptation metrics
        """
        concept_data = {
            "epoch": epoch,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "concept_similarity": concept_similarity,
            "prior_preservation_score": prior_preservation_score,
            "text_encoder_adaptation": text_encoder_adaptation
        }
        
        # Convert tensors to lists for JSON serialization
        if concept_embeddings is not None:
            concept_data["concept_embeddings"] = concept_embeddings.detach().cpu().tolist()
        if prior_embeddings is not None:
            concept_data["prior_embeddings"] = prior_embeddings.detach().cpu().tolist()
        
        self.log_data["concept_analysis"].append(concept_data)
        
        # Write concept analysis summary
        self._write_concept_summary(concept_data)
        
        # Save JSON backup
        with open(self.json_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
    
    def _write_concept_summary(self, concept_data: Dict[str, Any]):
        """Write concept analysis summary to text file."""
        epoch = concept_data["epoch"]
        concept_sim = concept_data.get("concept_similarity", 0)
        prior_pres = concept_data.get("prior_preservation_score", 0)
        
        concept_line = f"Concept Analysis Epoch {epoch:3d} | Similarity: {concept_sim:.3f} | Prior Pres: {prior_pres:.3f}"
        
        with open(self.log_file, 'a') as f:
            f.write(concept_line + "\n")
    
    def log_system_performance(self, 
                             epoch: int,
                             memory_usage: float,
                             gpu_utilization: Optional[float] = None,
                             cpu_utilization: Optional[float] = None,
                             training_speed: Optional[float] = None):
        """
        Log system performance metrics.
        
        Args:
            epoch: Current epoch
            memory_usage: GPU memory usage in GB
            gpu_utilization: GPU utilization percentage
            cpu_utilization: CPU utilization percentage
            training_speed: Steps per second
        """
        performance_data = {
            "epoch": epoch,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "memory_usage": memory_usage,
            "gpu_utilization": gpu_utilization,
            "cpu_utilization": cpu_utilization,
            "training_speed": training_speed
        }
        
        self.log_data["system_performance"].append(performance_data)
        
        # Write performance summary
        self._write_performance_summary(performance_data)
        
        # Save JSON backup
        with open(self.json_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
    
    def _write_performance_summary(self, performance_data: Dict[str, Any]):
        """Write system performance summary to text file."""
        epoch = performance_data["epoch"]
        memory = performance_data["memory_usage"]
        gpu_util = performance_data.get("gpu_utilization", 0)
        cpu_util = performance_data.get("cpu_utilization", 0)
        speed = performance_data.get("training_speed", 0)
        
        perf_line = f"Performance Epoch {epoch:3d} | Memory: {memory:.1f}GB | GPU: {gpu_util:.1f}% | CPU: {cpu_util:.1f}% | Speed: {speed:.1f} steps/s"
        
        with open(self.log_file, 'a') as f:
            f.write(perf_line + "\n")
    
    def log_final_summary(self, 
                         total_epochs: int,
                         total_steps: int,
                         total_time: float,
                         final_loss: float,
                         best_val_loss: float,
                         model_path: str,
                         training_config: Optional[Dict] = None,
                         final_metrics: Optional[Dict] = None,
                         spectral_analysis: Optional[Dict] = None,
                         performance_stats: Optional[Dict] = None,
                         concept_learning_summary: Optional[Dict] = None):
        """
        Create a comprehensive training summary file with all information needed to assess performance.
        
        This creates a single text file that contains everything needed to understand:
        - Training configuration and hyperparameters
        - Final model performance metrics
        - Spectral fidelity analysis
        - Concept learning progress
        - System performance statistics
        - Model architecture details
        - Training dynamics and convergence
        
        Args:
            total_epochs: Total number of epochs completed
            total_steps: Total number of training steps
            total_time: Total training time in seconds
            final_loss: Final training loss
            best_val_loss: Best validation loss achieved
            model_path: Path where final model was saved
            training_config: Complete training configuration
            final_metrics: Final epoch metrics
            spectral_analysis: Spectral fidelity analysis results
            performance_stats: System performance statistics
            concept_learning_summary: Concept learning analysis
        """
        summary_file = os.path.join(self.output_dir, "training_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DREAMBOOTH MULTISPECTRAL TRAINING SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            # Training Overview
            f.write("TRAINING OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Epochs: {total_epochs}\n")
            f.write(f"Total Steps: {total_steps}\n")
            f.write(f"Final Training Loss: {final_loss:.6f}\n")
            f.write(f"Best Validation Loss: {best_val_loss:.6f}\n")
            f.write(f"Model Saved To: {model_path}\n\n")
            
            # Training Configuration
            if training_config:
                f.write("TRAINING CONFIGURATION\n")
                f.write("-" * 40 + "\n")
                for key, value in training_config.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Final Metrics
            if final_metrics:
                f.write("FINAL METRICS\n")
                f.write("-" * 40 + "\n")
                for key, value in final_metrics.items():
                    if isinstance(value, (list, tuple)):
                        f.write(f"{key}: {[f'{v:.4f}' for v in value]}\n")
                    else:
                        f.write(f"{key}: {value:.6f}\n")
                f.write("\n")
            
            # Spectral Analysis
            if spectral_analysis:
                f.write("SPECTRAL FIDELITY ANALYSIS\n")
                f.write("-" * 40 + "\n")
                for key, value in spectral_analysis.items():
                    if isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for subkey, subvalue in value.items():
                            f.write(f"  {subkey}: {subvalue:.4f}\n")
                    else:
                        f.write(f"{key}: {value:.4f}\n")
                f.write("\n")
            
            # Concept Learning Summary
            if concept_learning_summary:
                f.write("CONCEPT LEARNING SUMMARY\n")
                f.write("-" * 40 + "\n")
                for key, value in concept_learning_summary.items():
                    if isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for subkey, subvalue in value.items():
                            f.write(f"  {subkey}: {subvalue:.4f}\n")
                    else:
                        f.write(f"{key}: {value:.4f}\n")
                f.write("\n")
            
            # Performance Statistics
            if performance_stats:
                f.write("SYSTEM PERFORMANCE STATISTICS\n")
                f.write("-" * 40 + "\n")
                for key, value in performance_stats.items():
                    if isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for subkey, subvalue in value.items():
                            f.write(f"  {subkey}: {subvalue:.2f}\n")
                    else:
                        f.write(f"{key}: {value:.2f}\n")
                f.write("\n")
            
            # Training Dynamics Analysis
            f.write("TRAINING DYNAMICS ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            # Analyze loss progression from JSON data
            if os.path.exists(self.json_file):
                try:
                    with open(self.json_file, 'r') as json_f:
                        training_data = json.load(json_f)
                    
                    if 'steps' in training_data and training_data['steps']:
                        steps = training_data['steps']
                        losses = [step.get('loss', 0) for step in steps if 'loss' in step]
                        
                        if losses:
                            f.write(f"Loss Progression:\n")
                            f.write(f"  Initial Loss: {losses[0]:.6f}\n")
                            f.write(f"  Final Loss: {losses[-1]:.6f}\n")
                            f.write(f"  Loss Reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%\n")
                            f.write(f"  Min Loss: {min(losses):.6f}\n")
                            f.write(f"  Max Loss: {max(losses):.6f}\n")
                            
                            # Convergence analysis
                            if len(losses) > 10:
                                recent_losses = losses[-10:]
                                recent_std = np.std(recent_losses)
                                f.write(f"  Recent Loss Std (last 10): {recent_std:.6f}\n")
                                if recent_std < 0.001:
                                    f.write("  ✓ Training appears to have converged\n")
                                else:
                                    f.write("  ⚠ Training may not have fully converged\n")
                except Exception as e:
                    f.write(f"Could not analyze training dynamics: {e}\n")
            
            # Model Assessment
            f.write("\nMODEL ASSESSMENT\n")
            f.write("-" * 40 + "\n")
            
            # Loss-based assessment
            if final_loss < 0.01:
                f.write("✓ Excellent training loss (< 0.01)\n")
            elif final_loss < 0.05:
                f.write("✓ Good training loss (< 0.05)\n")
            elif final_loss < 0.1:
                f.write("⚠ Acceptable training loss (< 0.1)\n")
            else:
                f.write("⚠ High training loss (> 0.1) - consider longer training\n")
            
            # Validation loss assessment
            if best_val_loss < 0.01:
                f.write("✓ Excellent validation loss (< 0.01)\n")
            elif best_val_loss < 0.05:
                f.write("✓ Good validation loss (< 0.05)\n")
            elif best_val_loss < 0.1:
                f.write("⚠ Acceptable validation loss (< 0.1)\n")
            else:
                f.write("⚠ High validation loss (> 0.1) - potential overfitting\n")
            

            
            # Recommendations
            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            
            if final_loss > 0.05:
                f.write("- Consider increasing training epochs\n")
                f.write("- Try adjusting learning rate\n")
                f.write("- Check data quality and preprocessing\n")
            
            if best_val_loss > 0.05:
                f.write("- Model may be overfitting - consider regularization\n")
                f.write("- Check validation data quality\n")
                f.write("- Consider early stopping with lower patience\n")
            

            
            f.write("\n- Monitor spectral fidelity in generated images\n")
            f.write("- Validate concept learning with test prompts\n")
            f.write("- Check for spectral signature preservation\n")
            
            # File locations
            f.write("\nIMPORTANT FILES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Training Summary: {summary_file}\n")
            f.write(f"Training Log: {self.log_file}\n")
            f.write(f"Training Data: {self.json_file}\n")
            f.write(f"Final Model: {model_path}\n")
            f.write(f"Best Model: {os.path.join(self.output_dir, 'best_model')}\n")
            f.write(f"Checkpoints: {os.path.join(self.output_dir, 'checkpoint-*')}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF TRAINING SUMMARY\n")
            f.write("=" * 80 + "\n")
        
        # Also log to main log file
        with open(self.log_file, 'a') as f:
            f.write(f"\nTraining completed successfully!\n")
            f.write(f"Final loss: {final_loss:.6f}, Best val loss: {best_val_loss:.6f}\n")
            f.write(f"Training summary saved to: {summary_file}\n")
        
        # Log to wandb if available
        if wandb.run:
            try:
                wandb.log({
                    "training_summary/final_loss": final_loss,
                    "training_summary/best_val_loss": best_val_loss,
                    "training_summary/total_epochs": total_epochs,
                    "training_summary/total_steps": total_steps,
                })
                
                # Upload summary file to wandb
                wandb.save(summary_file)
                
            except Exception as e:
                print(f"Warning: Failed to log final summary to wandb: {e}")
        
        print(f"Training summary saved to: {summary_file}")
    
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
    try:
        # Create a descriptive run name
        run_name = f"dreambooth_multispectral_{instance_prompt.replace(' ', '_')}"
        if class_prompt:
            run_name += f"_vs_{class_prompt.replace(' ', '_')}"
        run_name += f"_lr{args.learning_rate}_bs{args.train_batch_size}"
        
        wandb.init(
            project="dreambooth-multispectral",
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
    if not wandb.run:
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