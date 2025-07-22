"""
Training Logger for Multispectral VAE

This module provides comprehensive logging functionality for the multispectral VAE training process.
It captures key metrics, model health indicators, and training behavior in a compressed format
for quick analysis and debugging.

Features:
- Epoch-by-epoch training metrics
- Model health monitoring
- Spectral-specific indicators
- System information
- Compressed log format for easy parsing

SCIENTIFIC UTILITY:
------------------
This logger addresses critical needs in multispectral VAE training:

1. SPECTRAL FIDELITY MONITORING:
   - Band importance analysis reveals which spectral bands the model prioritizes
   - Enables detection of problematic bands that may be underweighted by attention mechanisms
   - Provides interpretable insights into spectral signature preservation

2. OUTPUT RANGE COMPLIANCE:
   - Post-adapter range statistics verify adherence to expected [-1, 1] output ranges
   - Quantifies range violations that can affect downstream pipeline compatibility
   - Enables systematic debugging of spectral transformation issues

3. TRAINING STABILITY ASSESSMENT:
   - Gradient clipping detection identifies optimization instability
   - Global scale monitoring tracks convergence of spectral scaling parameters
   - Comprehensive loss tracking enables multi-objective optimization analysis

4. REPRODUCIBILITY AND ANALYSIS:
   - Structured JSON logging enables automated post-training analysis
   - Compressed text format provides human-readable quick reference
   - Complete metric history supports scientific validation and comparison

DESIGN RATIONALE:
-----------------
The logger employs a dual-format approach (text + JSON) to balance human readability
with machine-processable data. This design supports both real-time monitoring during
training and comprehensive post-hoc analysis for scientific publication.

The spectral-specific metrics address unique challenges in multispectral imaging:
- Non-uniform band importance due to varying signal quality and biological relevance
- Complex spectral transformations that may violate expected output ranges
- Multi-objective loss functions requiring careful balance between spatial and spectral fidelity
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import torch
import numpy as np
from pathlib import Path


class TrainingLogger:
    """
    Comprehensive training logger for multispectral VAE training.
    
    Captures and logs:
    - Training progress metrics
    - Model health indicators
    - Validation performance
    - Spectral-specific metrics
    - System information
    
    SCIENTIFIC CONTRIBUTION:
    ------------------------
    This logger addresses the unique challenges of multispectral VAE training by providing:
    
    1. SPECTRAL ATTENTION ANALYSIS:
       - Monitors band importance weights from spectral attention mechanisms
       - Detects problematic bands that may be underweighted during training
       - Provides interpretable patterns of spectral focus (uniform/focused/balanced)
       - Enables validation of spectral signature preservation strategies
    
    2. OUTPUT RANGE VERIFICATION:
       - Quantifies compliance with expected [-1, 1] output ranges
       - Identifies systematic range violations that affect downstream compatibility
       - Provides detailed violation statistics for debugging spectral transformations
       - Supports assessment of spectral fidelity preservation
    
    3. TRAINING STABILITY MONITORING:
       - Tracks gradient clipping events that indicate optimization instability
       - Monitors global scale parameters for convergence analysis
       - Provides comprehensive loss decomposition for multi-objective optimization
       - Enables early detection of training divergence or collapse
    
    IMPLEMENTATION DESIGN:
    ----------------------
    The logger employs a hierarchical data structure that separates:
    - Epoch-level metrics (training progress, validation performance)
    - Model health indicators (numerical stability, parameter behavior)
    - Spectral-specific analysis (band importance, range compliance)
    - System-level information (timing, resource usage)
    
    This design enables both real-time monitoring and comprehensive post-training analysis.
    """
    
    def __init__(self, output_dir: str, model_name: str = "multispectral_vae"):
        """
        Initialize the training logger.
        
        Args:
            output_dir: Directory to save log files
            model_name: Name for the model/experiment
        """
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.log_dir = self.output_dir / "training_logs"
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
            "epochs": [],
            "model_health": [],
            "best_metrics": {},
            "final_summary": {}
        }
        
        # Track best metrics
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Write header
        self._write_header()
    
    def _write_header(self):
        """Write header information to the log file."""
        header = f"""
{'='*80}
MULTISPECTRAL VAE TRAINING LOG
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
    
    def log_epoch(self, 
                  epoch: int,
                  train_losses: Dict[str, float],
                  val_losses: Dict[str, float],
                  band_importance: Optional[Dict] = None,
                  band_importance_analysis: Optional[Dict] = None,
                  ssim_per_band: Optional[list] = None,
                  global_scale: Optional[float] = None,
                  learning_rate: Optional[float] = None,
                  grad_norm: Optional[float] = None,
                  gradient_clipping: Optional[bool] = None,
                  global_min_max_per_band: Optional[Dict] = None,
                  output_range_stats: Optional[Dict] = None,
                  post_adapter_range_stats: Optional[Dict] = None,
                  recon_mean_spectrum: Optional[list] = None):
        """
        Log epoch-level metrics.
        
        This method captures comprehensive training metrics including spectral-specific
        indicators that are critical for multispectral VAE training analysis.
        
        SPECTRAL-SPECIFIC METRICS:
        --------------------------
        
        1. BAND IMPORTANCE ANALYSIS:
           - band_importance: Raw attention weights from spectral attention mechanisms
           - band_importance_analysis: Processed analysis including problematic band detection
           - This enables assessment of whether the model is appropriately weighting
             different spectral bands based on their biological relevance and signal quality
        
        2. OUTPUT RANGE VERIFICATION:
           - post_adapter_range_stats: Detailed analysis of output range compliance
           - Quantifies violations of expected [-1, 1] ranges that affect downstream compatibility
           - Provides violation statistics for debugging spectral transformation issues
        
        3. SPECTRAL FIDELITY INDICATORS:
           - ssim_per_band: Structural similarity per spectral band
           - global_min_max_per_band: Range statistics per band
           - These metrics enable assessment of spectral signature preservation
           reconstructed mean spectrum
        
        TRAINING STABILITY METRICS:
        ---------------------------
        - gradient_clipping: Indicates optimization instability
        - global_scale: Convergence of spectral scaling parameters
        - grad_norm: Gradient magnitude for stability assessment
        
        Args:
            epoch: Current epoch number
            train_losses: Training loss dictionary
            val_losses: Validation loss dictionary
            band_importance: Spectral band importance weights
            band_importance_analysis: Detailed band importance analysis with problematic band detection
            ssim_per_band: SSIM scores per band
            global_scale: Global scale parameter value
            learning_rate: Current learning rate
            grad_norm: Gradient norm
            gradient_clipping: Whether gradient clipping occurred
            global_min_max_per_band: Global min/max values per spectral band
            output_range_stats: Output range statistics
            post_adapter_range_stats: Range distribution after output adapter (for [-1,1] verification)
        """
        epoch_data = {
            "epoch": epoch,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "learning_rate": learning_rate,
            "grad_norm": grad_norm,
            "gradient_clipping": gradient_clipping,
            "global_scale": global_scale,
        }
        if band_importance:
            epoch_data["band_importance"] = band_importance
        if band_importance_analysis:
            epoch_data["band_importance_analysis"] = band_importance_analysis
        if ssim_per_band:
            epoch_data["ssim_per_band"] = ssim_per_band
        if global_min_max_per_band:
            epoch_data["global_min_max_per_band"] = global_min_max_per_band
        if output_range_stats:
            epoch_data["output_range_stats"] = output_range_stats
        if post_adapter_range_stats:
            epoch_data["post_adapter_range_stats"] = post_adapter_range_stats
        if recon_mean_spectrum is not None:
            epoch_data["recon_mean_spectrum"] = recon_mean_spectrum
            # Optionally, compute deviation from reference if available
            ref = self.log_data.get("training_config", {}).get("reference_signature")
            if ref is not None:
                deviation = [float(r) - float(s) for r, s in zip(recon_mean_spectrum, ref)]
                epoch_data["recon_mean_spectrum_deviation"] = deviation
        self.log_data["epochs"].append(epoch_data)
        
        # Check for best model
        if val_losses.get('total_loss', float('inf')) < self.best_val_loss:
            self.best_val_loss = val_losses['total_loss']
            self.best_epoch = epoch
            self.log_data["best_metrics"] = {
                "best_epoch": epoch,
                "best_val_loss": self.best_val_loss,
                "train_loss_at_best": train_losses.get('total_loss', 0)
            }
        
        # Write to text file
        self._write_epoch_summary(epoch_data)
        # Save JSON backup
        with open(self.json_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
    
    def _write_epoch_summary(self, epoch_data: Dict[str, Any]):
        """Write epoch summary to text file."""
        epoch = epoch_data["epoch"]
        train_loss = epoch_data["train_losses"].get('total_loss', 0)
        val_loss = epoch_data["val_losses"].get('total_loss', 0)
        lr = epoch_data.get("learning_rate", 0)
        grad_norm = epoch_data.get("grad_norm", 0)
        gradient_clipping = epoch_data.get("gradient_clipping", False)
        global_scale = epoch_data.get("global_scale", 0)
        
        # Format SSIM
        ssim_str = ""
        if "ssim_per_band" in epoch_data:
            ssim_vals = epoch_data["ssim_per_band"]
            avg_ssim = np.mean(ssim_vals) if ssim_vals else 0
            ssim_str = f" | SSIM: {avg_ssim:.4f} [{', '.join([f'{v:.4f}' for v in ssim_vals])}]"
        
        # Format band importance
        band_importance_str = ""
        if "band_importance" in epoch_data:
            bi = epoch_data["band_importance"]
            band_importance_str = f" | Band Imp: [{', '.join([f'{v:.3f}' for v in bi.values()])}]"
        
        # Format band importance analysis (problematic band detection)
        band_analysis_str = ""
        if "band_importance_analysis" in epoch_data:
            analysis = epoch_data["band_importance_analysis"]
            if analysis.get("problematic_bands"):
                problematic = analysis["problematic_bands"]
                band_analysis_str = f" | Prob Bands: {problematic}"
            if analysis.get("attention_pattern"):
                pattern = analysis["attention_pattern"]
                band_analysis_str += f" | Pattern: {pattern}"
        
        # Format global min/max per band
        min_max_str = ""
        if "global_min_max_per_band" in epoch_data:
            min_max_data = epoch_data["global_min_max_per_band"]
            min_max_str = f" | Min/Max: [{', '.join([f'[{v[0]:.3f},{v[1]:.3f}]' for v in min_max_data.values()])}]"
        
        # Format output range
        range_str = ""
        if "output_range_stats" in epoch_data:
            stats = epoch_data["output_range_stats"]
            range_str = f" | Range: [{stats.get('global_min', 0):.3f}, {stats.get('global_max', 0):.3f}]"
        
        # Format post-adapter range verification
        post_range_str = ""
        if "post_adapter_range_stats" in epoch_data:
            post_stats = epoch_data["post_adapter_range_stats"]
            min_val = post_stats.get('min', 0)
            max_val = post_stats.get('max', 0)
            violations = post_stats.get('violations', 0)
            violation_pct = post_stats.get('violation_percentage', 0)
            compliance_95 = post_stats.get('compliance_95_percent', False)
            compliance_95_val = post_stats.get('compliance_95_percent_value', 0)
            
            # Color-code based on violations and 95% compliance
            if violations > 0:
                compliance_indicator = "✓95%" if compliance_95 else "✗95%"
                post_range_str = f" | Post-Adapter: [{min_val:.3f},{max_val:.3f}] | VIOLATIONS: {violations} ({violation_pct:.1f}%) | {compliance_indicator} ({compliance_95_val:.1f}%)"
            else:
                post_range_str = f" | Post-Adapter: [{min_val:.3f},{max_val:.3f}] ✓ | 95%: ✓ ({compliance_95_val:.1f}%)"
        
        # Format gradient clipping indicator
        clip_str = " [CLIP]" if gradient_clipping else ""
        
        # Main epoch line
        epoch_line = f"Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.2e} | Grad: {grad_norm:.3f}{clip_str} | Scale: {global_scale:.4f}{ssim_str}{band_importance_str}{band_analysis_str}{min_max_str}{range_str}{post_range_str}"
        
        # Add best model indicator
        if epoch == self.best_epoch:
            epoch_line += " [BEST]"
        
        with open(self.log_file, 'a') as f:
            f.write(epoch_line + "\n")
        
        # Add mean spectrum info if present
        if "recon_mean_spectrum" in epoch_data:
            rms = epoch_data["recon_mean_spectrum"]
            rms_str = f"Reconstructed mean spectrum: {[f'{v:.4f}' for v in rms]}"
            with open(self.log_file, 'a') as f:
                f.write(rms_str + "\n")
        if "recon_mean_spectrum_deviation" in epoch_data:
            dev = epoch_data["recon_mean_spectrum_deviation"]
            dev_str = f"Deviation from reference: {[f'{v:+.4f}' for v in dev]}"
            with open(self.log_file, 'a') as f:
                f.write(dev_str + "\n")
    
    def log_model_health(self, health_data: Dict[str, Any]):
        """Log model health indicators."""
        self.log_data["model_health"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **health_data
        })
        
        health_str = f"""
MODEL HEALTH CHECK:
{'-'*20}
"""
        for key, value in health_data.items():
            health_str += f"{key}: {value}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(health_str + "\n")
    
    def log_final_summary(self, 
                         total_epochs: int,
                         total_time: float,
                         final_train_loss: float,
                         final_val_loss: float,
                         model_path: str):
        """Log final training summary."""
        summary = {
            "total_epochs": total_epochs,
            "total_time": total_time,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "model_path": model_path,
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.log_data["final_summary"] = summary
        
        summary_str = f"""
{'='*80}
TRAINING SUMMARY
{'='*80}
Total Epochs: {total_epochs}
Total Time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)
Final Train Loss: {final_train_loss:.6f}
Final Val Loss: {final_val_loss:.6f}
Best Epoch: {self.best_epoch}
Best Val Loss: {self.best_val_loss:.6f}
Model Saved: {model_path}
End Time: {summary['end_time']}
{'='*80}
"""
        
        with open(self.log_file, 'a') as f:
            f.write(summary_str)
        
        # Save final JSON
        with open(self.json_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
    
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


def create_training_logger(output_dir: str, model_name: str = "multispectral_vae") -> TrainingLogger:
    """
    Factory function to create a training logger.
    
    Args:
        output_dir: Directory to save log files
        model_name: Name for the model/experiment
    
    Returns:
        TrainingLogger instance
    """
    return TrainingLogger(output_dir, model_name)


def analyze_band_importance(model, problematic_bands=None):
    """
    Analyze band importance and detect problematic bands.
    
    This function provides comprehensive analysis of spectral attention mechanisms
    in multispectral VAE training. It addresses the critical need to understand
    how the model weights different spectral bands and whether problematic bands
    are being appropriately handled.
    
    SCIENTIFIC UTILITY:
    -------------------
    
    1. PROBLEMATIC BAND DETECTION:
       - Identifies bands that may be underweighted by attention mechanisms
       - Enables validation of spectral signature preservation strategies
       - Provides early warning of potential spectral fidelity issues
    
    2. ATTENTION PATTERN ANALYSIS:
       - Classifies attention patterns as uniform, focused, or balanced
       - Enables assessment of spectral focus strategies
       - Supports optimization of spectral attention mechanisms
    
    3. SPECTRAL INTERPRETABILITY:
       - Maps attention weights to biological wavelengths
       - Enables scientific validation of band importance
       - Supports plant health analysis through spectral signature interpretation
    
    IMPLEMENTATION DETAILS:
    -----------------------
    The function analyzes attention weights from spectral attention mechanisms
    and applies statistical analysis to detect patterns and anomalies:
    
    - Mean and standard deviation analysis for pattern classification
    - Threshold-based detection of underweighted bands
    - Wavelength mapping for biological interpretability
    - Statistical significance assessment for attention patterns
    
    Args:
        model: The multispectral VAE model
        problematic_bands: List of band indices known to be problematic (e.g., [2, 4] for bands 3 and 5)
    
    Returns:
        Dictionary with band importance analysis containing:
        - band_importance: Raw attention weights mapped to wavelengths
        - mean_importance: Statistical mean of attention weights
        - std_importance: Standard deviation of attention weights
        - problematic_bands: List of detected problematic bands with weights
        - attention_pattern: Classification of attention pattern (uniform/focused/balanced)
        - lowest_band: Band with lowest attention weight
        - highest_band: Band with highest attention weight
    """
    # Spectral attention has been removed - band importance analysis is no longer available
    # Return None to indicate band importance analysis is disabled
    return None
    
    # Convert to list for analysis
    importance_values = list(band_importance.values())
    band_indices = list(band_importance.keys())
    
    # Analyze attention patterns
    mean_importance = np.mean(importance_values)
    std_importance = np.std(importance_values)
    
    # Detect problematic bands (low attention)
    problematic_detected = []
    if problematic_bands:
        for band_idx in problematic_bands:
            if band_idx < len(importance_values):
                if importance_values[band_idx] < mean_importance - std_importance:
                    problematic_detected.append(f"B{band_idx+1}({importance_values[band_idx]:.3f})")
    
    # Determine attention pattern
    if std_importance < 0.1:
        pattern = "uniform"
    elif max(importance_values) > mean_importance + 2*std_importance:
        pattern = "focused"
    else:
        pattern = "balanced"
    
    return {
        "band_importance": band_importance,
        "mean_importance": mean_importance,
        "std_importance": std_importance,
        "problematic_bands": problematic_detected,
        "attention_pattern": pattern,
        "lowest_band": f"B{np.argmin(importance_values)+1}({min(importance_values):.3f})",
        "highest_band": f"B{np.argmax(importance_values)+1}({max(importance_values):.3f})"
    }


def compute_post_adapter_range_stats(model_output):
    """
    Compute range statistics after the output adapter to verify [-1, 1] compliance.
    
    This function addresses the critical issue of output range compliance in multispectral
    VAE training. It provides comprehensive analysis of whether model outputs adhere to
    expected [-1, 1] ranges that are required for downstream pipeline compatibility.
    
    SCIENTIFIC UTILITY:
    -------------------
    
    1. RANGE COMPLIANCE VERIFICATION:
       - Quantifies violations of expected [-1, 1] output ranges
       - Enables systematic debugging of spectral transformation issues
       - Provides violation statistics for downstream pipeline assessment
    
    2. SPECTRAL FIDELITY ASSESSMENT:
       - Analyzes whether spectral transformations preserve expected ranges
       - Identifies systematic range violations that affect spectral fidelity
       - Supports optimization of spectral transformation mechanisms
    
    3. DOWNSTREAM COMPATIBILITY:
       - Ensures compatibility with standard image processing pipelines
       - Identifies range violations that may affect model integration
       - Provides metrics for post-processing optimization
    
    IMPLEMENTATION DETAILS:
    -----------------------
    The function performs comprehensive statistical analysis of output ranges:
    
    - Basic statistics (min, max, mean, std) for range assessment
    - Violation counting and percentage calculation
    - Severity analysis (below minimum vs above maximum violations)
    - Maximum violation magnitude for severity assessment
    - 95% range compliance check for practical pipeline compatibility
    
    This analysis enables both real-time monitoring during training and
    comprehensive post-training assessment of spectral transformation quality.
    
    Args:
        model_output: Output tensor from the model (after output adapter)
    
    Returns:
        Dictionary with range statistics and violation analysis containing:
        - min/max: Minimum and maximum values in the output
        - mean/std: Statistical measures of output distribution
        - violations: Number of pixels outside [-1, 1] range
        - violation_percentage: Percentage of pixels violating range
        - below_min/above_max: Separate counts for below -1 and above 1 violations
        - max_violation: Maximum magnitude of range violation
        - in_range: Boolean indicating complete range compliance
        - compliance_95_percent: Boolean indicating if 95%+ of outputs are within [-1, 1]
        - compliance_95_percent_value: Actual percentage of outputs within [-1, 1]
    """
    if model_output is None:
        return None
    
    # Convert to numpy for analysis
    if torch.is_tensor(model_output):
        output_np = model_output.detach().cpu().numpy()
    else:
        output_np = model_output
    
    # Compute basic statistics
    min_val = float(np.min(output_np))
    max_val = float(np.max(output_np))
    mean_val = float(np.mean(output_np))
    std_val = float(np.std(output_np))
    
    # Count violations of [-1, 1] range
    violations = np.sum((output_np < -1.0) | (output_np > 1.0))
    total_pixels = output_np.size
    violation_percentage = (violations / total_pixels) * 100 if total_pixels > 0 else 0
    
    # Calculate 95% compliance
    compliance_95_percent_value = 100.0 - violation_percentage
    compliance_95_percent = compliance_95_percent_value >= 95.0
    
    # Analyze violation severity
    if violations > 0:
        below_min = np.sum(output_np < -1.0)
        above_max = np.sum(output_np > 1.0)
        max_violation = max(abs(min_val + 1.0), abs(max_val - 1.0))
    else:
        below_min = 0
        above_max = 0
        max_violation = 0.0
    
    return {
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "std": std_val,
        "violations": int(violations),
        "violation_percentage": violation_percentage,
        "below_min": int(below_min),
        "above_max": int(above_max),
        "max_violation": max_violation,
        "in_range": violations == 0,
        "compliance_95_percent": compliance_95_percent,
        "compliance_95_percent_value": compliance_95_percent_value
    } 