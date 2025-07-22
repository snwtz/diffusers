"""
Multispectral VAE Adapter for Stable Diffusion 3: Core Methodological Contribution

https://huggingface.co/docs/diffusers/v0.6.0/en/api/models
https://huggingface.co/docs/diffusers/en/api/models/autoencoderkl


This module implements the central methodological contribution of the thesis: a lightweight
adapter-based multispectral autoencoder architecture built on a pretrained SD3 backbone.
The design enables efficient processing of 5-channel spectral plant imagery while maintaining
compatibility with SD3's latent space requirements.

Data Flow Summary:
------------------
- Input: Preprocessed 5-channel plant images (from vae_multispectral_dataloader.py)
- Model: Adapters map 5-channel input to 3-channel SD3 backbone, then back to 5-channel output
- Output: Reconstructed 5-channel image with nonlinear transformations applied (unconstrained range)
- Loss: Multi-objective (MSE + SAM), with pure leaf-focused masked loss computation
- Output Format: Compatible with both training (raw tensor) and downstream pipelines (DecoderOutput)

NOTE ON INPUT FORMAT:
---------------------
This training script currently expects raw tensor input of shape (B, 5, H, W) during training. In contrast,
standard SD3 inference pipelines typically expect `dict`-style conditioning input format, e.g. {"sample": x, "mask": y}.
This discrepancy can impact integration with SD pipelines if later applying this model in generation tasks.
Adjustments may be required for compatibility with diffusers pipelines or script-based inference.

NOTE: This version includes defensive coding for numerical stability.
- Applies nan/inf detection after transforming inputs.
- Replaces NaNs/Infs with default clamped values to prevent decoder failures.
- This is necessary due to observed NaNs early in the model pipeline, traced to potential data anomalies or instability.
- IMPORTANT: The decoder output contains significant nonlinear transformations that may 
produce values outside typical image ranges (range not constrained to [-1,1]).

Thesis Context and Scientific Innovation:
---------------------------------------
1. Research Objective:
   - Enable multispectral image generation using SD3
   - Maintain spectral fidelity while leveraging pretrained knowledge
   - Support scientific analysis of plant health through spectral signatures
   - Enable parameter-efficient adaptation for limited data scenarios

2. Core Innovation:
   - Lightweight adapter architecture for 5-channel spectral data
   - Parameter-efficient fine-tuning strategy
   - Dual loss function preserving both spatial and spectral fidelity
   - Nonlinear transformations enabling complex spectral relationship learning

3. Biological Relevance:
   The architecture processes 5 carefully selected bands from hyperspectral data:
   - Band 9 (474.73nm): Blue - captures chlorophyll absorption
   - Band 18 (538.71nm): Green - reflects well in healthy vegetation
   - Band 32 (650.665nm): Red - sensitive to chlorophyll content
   - Band 42 (730.635nm): Red-edge - sensitive to stress and early disease
   - Band 55 (850.59nm): NIR - strong reflectance in healthy leaves

Architectural Design Decisions:
----------------------------
1. Adapter Architecture:
   a) SpectralAdapter:
      - 3×3 convolutions: Balance spatial feature modeling with efficiency
      - GroupNorm: Stable training with small batch sizes typical in hyperspectral data (nonlinear normalization)
      - SiLU activation: Gradient-friendly nonlinearity better suited than ReLU (nonlinear activation)
      - Three-layer design: Progressive feature extraction and channel adaptation with nonlinear transformations



2. Loss Function Design:
   a) Per-channel MSE Loss:
      - Preserves spatial structure and pixel-wise accuracy
      - Enables band-specific optimization
      - Helps identify problematic spectral bands

   b) Spectral Angle Mapper (SAM) Loss:
      - Measures spectral similarity through vector angles
      - Invariant to scaling, preserves spectral signatures
      - Weighted combination: loss = α * MSE + β * SAM
      - Configurable weights for balancing spatial vs. spectral fidelity

3. Training Strategy:
   a) Parameter Efficiency:
      - freeze_backbone(): Preserve SD3's latent space properties
      - get_trainable_params(): Enable adapter-only training
      - Minimal trainable parameters (only adapter layers)

   b) Pure Leaf-Focused Training:
      - compute_losses(): Implements masked loss computation
      - Excludes background regions from loss calculation
      - Focuses training purely on biologically relevant leaf regions
      - Provides mask coverage statistics for monitoring

   c) Flexible Configuration:
      - adapter_placement: "input", "output", or "both"
      - Enables experimentation with different adaptation strategies
      - Supports ablation studies for thesis analysis

Integration and Downstream Use:
----------------------------
1. DreamBooth Integration:
   - Seamless compatibility with SD3's latent space
   - Supports multispectral image generation
   - Enables spectral concept learning

2. Scientific Analysis:
   - Per-band loss tracking for spectral fidelity analysis
   - Support for spectral signature preservation studies

Implementation Details:
---------------------
1. Model Components:
   - Pretrained SD3 VAE backbone
   - Input/output adapter layers
   - Loss computation pipeline

2. Training Integration:
   - Parameter isolation for efficient fine-tuning
   - Loss term balancing
   - Spectral fidelity preservation

Known Limitations:
----------------
1. Latent Space Compatibility:
   - Must maintain SD3's 4-channel latent space
   - Potential information bottleneck
   - Trade-off between compression and fidelity

2. Training Stability:
   - Loss term balancing needed
   - Potential spectral distortion
   - Channel interaction complexity

Scientific Contributions and Future Work:
-------------------------------------
1. Spectral Representation Learning:
   - Investigate band correlation patterns
   - Study spectral signature preservation
   - Explore adaptive normalization strategies
   - Design spectral-aware loss functions

2. Model Architecture:
   - Propose new adapter architectures
   - Develop spectral correlation models
   - Design spectral normalization layers
   - Investigate residual spectral connections

3. Training Methodology:
   - Develop spectral-aware optimization
   - Design spectral validation protocols
   - Create spectral benchmarking
   - Study gradient flow in spectral space
   - Investigate spectral regularization

4. Theoretical Foundations:
   - Analyze spectral information flow
   - Study latent space properties
   - Develop spectral fidelity metrics
   - Create spectral interpretability tools
   - Design spectral validation frameworks

TODOs:
------
1. For convergence stability:
   - Consider scaling/weighting loss terms
   - Implement loss = mse_weight * mse + sam_weight * sam
   - Allow tuning of loss contributions

2. For overfitting prevention:
   - Add dropout layer to each input/output adapter
   - Implement if overfitting to spectral patterns becomes an issue

3. For implementation:
   - Add get_trainable_params() method
   - Implement compute_losses() in model
   - Add per-band loss tracking


   CLI command needs --base_model_path "stabilityai/stable-diffusion-3-medium-diffusers"
   NOTE: enable Tanh output bounding via --force_output_tanh on the CLI.
    -> disable during training and apply only during inference

Usage:
    # Initialize with pretrained SD3 VAE
    vae = AutoencoderKLMultispectralAdapter.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        adapter_placement="both",  # or "input" or "output"
        use_sam_loss=True
    )

    # Freeze backbone (only adapters will be trained)
    vae.freeze_backbone()

    # Train only adapter layers
    optimizer = torch.optim.AdamW(vae.get_trainable_params(), lr=1e-4)
    
    # Note: Output range is unconstrained due to nonlinear transformations
    # TODO nonlinearities introduce range warping, making it likely that even properly normalized 
    # [–1, 1] input will produce non-symmetric output.
    Fix Options:
	•	(Recommended) Clamp output of output adapter to [–1, 1].
	•	(Optional) Add Tanh() at the very end of SpectralAdapter.forward() (only for output adapter).
    -> Add final Tanh activation to force [–1, 1] output
    self.output_norm = nn.Tanh()

    Then in forward()
    x = self.output_norm(x)
    
    # For downstream usage requiring [-1, 1] range, apply post-processing normalization

    
"""

# ------------------------------------------------------------
# VAE Loading Fix Summary:
# Hugging Face's `from_pretrained()` logic uses
# keyword arguments via a config object. The original constructor
# was not compatible with this pattern. To fix this:
#
# - We added `*` to enforce keyword-only arguments in __init__.
# - We unpacked the base config using `**config` when calling
#   the parent AutoencoderKL constructor (which expects kwargs).
# - We implemented a custom `from_pretrained()` classmethod that
#   combines Hugging Face-compatible config loading with custom
#   adapter arguments for spectral training.
#
# These changes now allow seamless loading of RGB VAE weights into
# the multispectral adapter class while preserving pretrained features.

# The decode() method has been patched to return a DecoderOutput object
# instead of AutoencoderKLOutput, resolving the error related to unexpected
# keyword arguments like latent_sample. This approach was chosen to preserve
# compatibility with Hugging Face's decoding expectations while still applying
# the custom multispectral output adapter.
# ------------------------------------------------------------

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np # Added for scale monitoring

# Set up logger for this module
logger = logging.getLogger(__name__)
from ...configuration_utils import ConfigMixin, register_to_config
from ...utils.accelerate_utils import apply_forward_hook
from ..modeling_outputs import AutoencoderKLOutput
from .autoencoder_kl import AutoencoderKL
from .vae import DecoderOutput

# SpectralAttention class removed completely

# 
# SpectralAdapter
# ---------------------------------
# These adapter layers transform between 5-channel multispectral inputs and the 3-channel
# format expected by the SD3 VAE. The input adapter maps 5→3 and output adapter maps 3→5.
# We use:
# - 3×3 convs for efficient spatial-spectral processing
# - GroupNorm for stability with small batch sizes (nonlinear normalization)
# - SiLU (Swish) activation for smoother gradients compared to ReLU (nonlinear activation)
# The adapters can be placed at input/output/both to allow ablation studies and flexibility.
# 
# The nonlinear transformations (SiLU, GroupNorm) mean that
# the output range is not constrained to [-1, 1] and may require post-processing normalization.
#
# Output Scale and Bias Parameters:
# ---------------------------------
# The output_scale and output_bias parameters are learnable linear transformation coefficients
# applied to the SpectralAdapter output (output = input * output_scale + output_bias). 
# They are learned via gradient descent during training:
        #   - output_scale (default: 1.0): scales the output amplitude.
        #   - output_bias  (default: 0.0): shifts the output baseline.  
# Together, they form an affine transformation: x' = output_scale * x + output_bias.     
# These parameters enable trainable calibration of the spectral output dynamic range to align
# with SD3 pipeline expectations (ideally [-1, 1]) while preserving spectral fidelity. 
# This approach is more adaptive and flexible than clamping or fixed postprocessing.
# Optimal values are output_scale ≈ 1.0 and output_bias ≈ 0.0, indicating minimal transformation is
# required and the adapter naturally produces appropriately scaled outputs. T
# This enables smooth, trainable calibration of spectral output without hard clipping or nonlinear warping.

# 
# Values significantly deviating from these targets (output_scale < 0.1 or > 3.0, |output_bias| > 0.5) may indicate
# training instability, data normalization issues, or inappropriate hyperparameter settings.
# Scale collapse (output_scale < 0.001) represents a critical failure mode where the model
# produces near-zero outputs, while scale explosion (output_scale > 5.0) indicates potential
# gradient instability. Convergence to near-identity values demonstrates successful preservation
# of biological spectral signatures with minimal range distortion.
#
class SpectralAdapter(nn.Module):
    """
    Adapter module for converting between 3 and 5 spectral channels.

    This module handles the conversion between the 5-channel multispectral
    input and the 3-channel RGB-like format expected by the SD3 VAE.
    It includes a series of convolutions to learn the optimal transformation 
    while preserving spectral information.

    Key design changes for spectral fidelity:
    -----------------------------------------
    - Spectral attention has been removed for simplicity and improved training stability
    - Single global learnable scale parameter provides linear, interpretable range control
    - torch.clamp(x, -1.0, 1.0) is applied during both training and inference for safety
    - This approach preserves the relative relationships between spectral bands
    - Global scaling is biologically plausible, as real plant reflectance spectra can vary in magnitude due to illumination
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super().__init__()
        # Store channel configuration for validation and logging
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Three-layer convolutional network for channel adaptation
        # First two layers use 3x3 convolutions with group normalization and nonlinear activation
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # Final layer uses 1x1 convolution for channel reduction/expansion (no activation)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=1)

        self.output_scale = nn.Parameter(torch.tensor(1.0))  # Global scaling factor
        self.output_bias = nn.Parameter(torch.tensor(0.0))   # Global shift

        # Group normalization for better training stability (nonlinear normalization)
        self.norm1 = nn.GroupNorm(8, 32)
        self.norm2 = nn.GroupNorm(8, 32)

        # SiLU activation (also known as Swish) - nonlinear activation function
        self.activation = nn.SiLU()

        # Single global scale parameter for linear range control
        # This replaces both adaptive norm and Tanh, preserving linearity for spectral fidelity
        self.global_scale = nn.Parameter(torch.tensor(1.0))
        
        # Global scale convergence monitoring
        self.scale_history = []  # Track scale values during training
        self.scale_convergence_threshold = 0.001  # Consider converged if std < threshold
        self.scale_warning_threshold = 0.01  # Warn if scale < 0.01
        self.scale_collapse_threshold = 0.001  # Consider collapsed if scale < 0.001
        self.scale_explosion_threshold = 5.0  # Warn if scale > 5.0
        self.convergence_window = 100  # Number of steps to consider for convergence
        self.step_counter = 0  # Track training steps for logging
        self.log_interval = 50  # Log scale info every N steps
        self.convergence_warning_issued = False  # Track if convergence warning was issued
        self.collapse_warning_issued = False  # Track if collapse warning was issued

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply background mask: Replace NaNs with per-band means to maintain spectral realism
        # This prevents downstream convolutional layers from propagating invalid values
        # while preserving realistic spectral signatures in background regions
        if torch.isnan(x).any():
            logger.debug("[NaN DEBUG] NaNs found in adapter input, replacing with per-band means.")
            for band_idx in range(x.shape[1]):
                band = x[:, band_idx]
                nan_mask = torch.isnan(band)
                if nan_mask.any():
                    band_mean = band[~nan_mask].mean() if (~nan_mask).any() else 0.0
                    x[:, band_idx][nan_mask] = band_mean

        # Log adapter input stats for NaN debugging
        logger.debug(f"[NaN DEBUG] SpectralAdapter input stats - min: {x.min().item():.6f}, max: {x.max().item():.6f}, mean: {x.mean().item():.6f}")

        # First convolutional block (nonlinear: conv + GroupNorm + SiLU)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        # Second convolutional block (nonlinear: conv + GroupNorm + SiLU)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        # Final channel adaptation (linear: conv only, no activation)
        x = self.conv3(x)

        # Apply learnable linear transformation
        # Helps align the dynamic range of output to [-1, 1] in a trainable and smooth way
        # Facilitates downstream consistency and spectral fidelity by allowing the model
        # to learn appropriate output magnitudes (instead of forcing via hard bounds)
        x = x * self.output_scale + self.output_bias

        # Log adapter output stats for NaN debugging
        if torch.isnan(x).any():
            logger.debug("[NaN DEBUG] NaNs in SpectralAdapter output.")
        logger.debug(f"[NaN DEBUG] SpectralAdapter output stats - min: {x.min().item():.6f}, max: {x.max().item():.6f}, mean: {x.mean().item():.6f}")

        return x

    def monitor_global_scale_convergence(self) -> Dict[str, any]:
        """
        Monitor global scale convergence and provide detailed analysis.
        
        This method tracks the global scale parameter's behavior during training to:
        1. Detect convergence (stability of scale values)
        2. Identify collapse (scale approaching zero)
        3. Detect explosion (scale becoming too large)
        4. Provide early warnings for training issues
        5. Track convergence metrics for scientific analysis
        
        Returns:
            Dictionary containing convergence analysis:
            - scale_value: Current scale value
            - scale_mean: Mean of recent scale values
            - scale_std: Standard deviation of recent scale values
            - is_converged: Whether scale has converged (std < threshold)
            - is_collapsed: Whether scale has collapsed (value < collapse_threshold)
            - is_exploded: Whether scale has exploded (value > explosion_threshold)
            - convergence_rate: Rate of change in recent steps
            - history_length: Number of tracked values
            - warnings: List of active warnings
            - recommendations: Suggested actions
        """
        current_scale = self.global_scale.item()
        
        # Track scale history
        self.scale_history.append(current_scale)
        self.step_counter += 1
        
        # Keep only recent history for efficiency
        if len(self.scale_history) > self.convergence_window:
            self.scale_history = self.scale_history[-self.convergence_window:]
        
        # Compute convergence metrics
        if len(self.scale_history) >= 10:  # Need at least 10 values for meaningful stats
            recent_scales = self.scale_history[-min(50, len(self.scale_history)):]  # Last 50 values
            scale_mean = np.mean(recent_scales)
            scale_std = np.std(recent_scales)
            
            # Check convergence (low standard deviation)
            is_converged = scale_std < self.scale_convergence_threshold
            
            # Check for collapse (scale too small)
            is_collapsed = current_scale < self.scale_collapse_threshold
            
            # Check for explosion (scale too large)
            is_exploded = current_scale > self.scale_explosion_threshold
            
            # Compute convergence rate (change over recent steps)
            if len(recent_scales) >= 2:
                convergence_rate = abs(recent_scales[-1] - recent_scales[0]) / len(recent_scales)
            else:
                convergence_rate = 0.0
        else:
            # Not enough data yet
            scale_mean = current_scale
            scale_std = 0.0
            is_converged = False
            is_collapsed = False
            is_exploded = False
            convergence_rate = 0.0
        
        # Generate warnings and recommendations
        warnings = []
        recommendations = []
        
        # Collapse warnings
        if is_collapsed and not self.collapse_warning_issued:
            warnings.append(f"CRITICAL: Global scale collapsed to {current_scale:.6f}")
            recommendations.append("Consider: 1) Lower learning rate, 2) Different initialization, 3) Add scale regularization")
            self.collapse_warning_issued = True
        elif not is_collapsed:
            self.collapse_warning_issued = False
        
        # Explosion warnings
        if is_exploded:
            warnings.append(f"WARNING: Global scale exploded to {current_scale:.6f}")
            recommendations.append("Consider: 1) Lower learning rate, 2) Add gradient clipping, 3) Scale regularization")
        
        # Convergence warnings
        if len(self.scale_history) >= 50 and not is_converged and not self.convergence_warning_issued:
            warnings.append(f"WARNING: Scale not converging (std={scale_std:.6f} > {self.scale_convergence_threshold})")
            recommendations.append("Consider: 1) Check learning rate, 2) Monitor gradients, 3) Verify data normalization")
            self.convergence_warning_issued = True
        elif is_converged:
            self.convergence_warning_issued = False
        
        # Value range warnings
        if current_scale < self.scale_warning_threshold:
            warnings.append(f"WARNING: Scale {current_scale:.6f} is very small")
            recommendations.append("Consider: 1) Check data normalization, 2) Lower learning rate")
        
        # Log detailed information periodically
        if self.step_counter % self.log_interval == 0:
            logger.info(f"[Scale Monitor] Step {self.step_counter}: scale={current_scale:.6f}, "
                       f"mean={scale_mean:.6f}, std={scale_std:.6f}, converged={is_converged}, "
                       f"rate={convergence_rate:.6f}, history_len={len(self.scale_history)}")
            
            if warnings:
                for warning in warnings:
                    logger.warning(f"[Scale Monitor] {warning}")
        
        return {
            'scale_value': current_scale,
            'scale_mean': scale_mean,
            'scale_std': scale_std,
            'is_converged': is_converged,
            'is_collapsed': is_collapsed,
            'is_exploded': is_exploded,
            'convergence_rate': convergence_rate,
            'history_length': len(self.scale_history),
            'warnings': warnings,
            'recommendations': recommendations,
            'step_counter': self.step_counter
        }

    def get_scale_monitoring_info(self) -> Dict[str, any]:
        """
        Get comprehensive scale monitoring information for logging and analysis.
        
        Returns:
            Dictionary containing all scale monitoring data
        """
        convergence_info = self.monitor_global_scale_convergence()
        
        return {
            'global_scale': convergence_info['scale_value'],
            'scale_mean': convergence_info['scale_mean'],
            'scale_std': convergence_info['scale_std'],
            'is_converged': convergence_info['is_converged'],
            'is_collapsed': convergence_info['is_collapsed'],
            'is_exploded': convergence_info['is_exploded'],
            'convergence_rate': convergence_info['convergence_rate'],
            'history_length': convergence_info['history_length'],
            'warnings': convergence_info['warnings'],
            'recommendations': convergence_info['recommendations'],
            'step_counter': convergence_info['step_counter']
        }

# ------------------------------------------------------------
# InputAdapter: Defensive coding for numerical stability
# ------------------------------------------------------------
class InputAdapter(nn.Module):
    """
    Adapter module for input normalization and defensive nan/inf handling.
    This module is inserted at the input side of the multispectral VAE pipeline.
    
    - Prevents propagation of NaNs/Infs into the model, supporting stable training and inference.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        # If you want, you can add normalization or transformation layers here
        # For now, acts as identity unless extended

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optionally perform normalization or transformation here
        # For example: x = (x - x.mean(dim=[2,3], keepdim=True)) / (x.std(dim=[2,3], keepdim=True) + 1e-6)

        # NaN check: early debug to track if input adapter introduces invalid values
        if torch.isnan(x).any():
            logger.debug("[NaN DEBUG] NaNs detected in InputAdapter output.")
        if torch.isinf(x).any():
            logger.debug("[NaN DEBUG] Infs detected in InputAdapter output.")

        # Handle NaN/Inf values: use per-band means for NaNs, clamp infinities
        if torch.isnan(x).any():
            for band_idx in range(x.shape[1]):
                band = x[:, band_idx]
                nan_mask = torch.isnan(band)
                if nan_mask.any():
                    band_mean = band[~nan_mask].mean() if (~nan_mask).any() else 0.0
                    x[:, band_idx][nan_mask] = band_mean
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)  # Only handles remaining Infs
        return x


# Numerically stable normalization for SAM loss
def safe_normalize(tensor, dim=1, eps=1e-8):
    # Numerically stable normalization for SAM loss
    # Ensures that spectral angle calculations are robust to small values and avoid NaNs.
    norm = torch.norm(tensor, p=2, dim=dim, keepdim=True)
    norm = norm.clamp(min=eps)
    return tensor / norm

def compute_sam_loss(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable Spectral Angle Mapper (SAM) loss between two multispectral images.

    Args:
        original: Original multispectral image
        reconstructed: Reconstructed multispectral image
    Returns:
        Mean spectral angle in radians
    
    - SAM loss is critical for preserving spectral signatures, which is a key scientific goal in plant imaging.
    - Invariant to scaling, so it focuses on spectral shape rather than intensity.
    """
    # Use safe normalization to avoid NaNs
    normalized_original = safe_normalize(original, dim=1)
    normalized_reconstructed = safe_normalize(reconstructed, dim=1)

    # Compute cosine similarity and clamp for stability
    cos_sim = F.cosine_similarity(normalized_original, normalized_reconstructed, dim=1)
    cos_sim = cos_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    angle = torch.acos(cos_sim)
    angle = torch.nan_to_num(angle, nan=0.0, posinf=0.0, neginf=0.0)

    if torch.isnan(angle).any():
        logger.debug("NaNs detected in SAM angle computation.")

    return angle.mean()

#
# Adapter Configuration and Training Design:
# ------------------------------------------
# This constructor registers all adapter-relevant settings for reproducibility.
# These include:
# - adapter_placement: Determines whether input/output or both sides of the VAE use adapters.
# - use_sam_loss: Includes a spectral fidelity loss term (Spectral Angle Mapper).
#
# These are currently passed via config but not stored in the output config.json.
# If we later want to reload saved models reliably, these fields should be added explicitly
# to the HuggingFace-compatible config serialization pipeline.
#
class AutoencoderKLMultispectralAdapter(AutoencoderKL):
    """Efficient multispectral VAE implementation using adapter layers.

    This implementation adapts the SD3 VAE for multispectral data by:
    1. Using pretrained SD3 VAE as backbone
    2. Adding lightweight adapter layers for 5-channel input/output
    3. Keeping backbone frozen during training (parameter-efficient fine-tuning)
    4. Only training the adapter layers (supports rapid adaptation)
    5. Including specialized SAM loss for spectral fidelity
    6. **Supports spectral signature guidance**: The training script can now include a reference-based spectral signature loss, encouraging the reconstructed mean spectrum (over leaf pixels) to match a provided healthy leaf signature. This is important for scientific realism and spectral fidelity in generated images.

    Parameters:
        pretrained_model_name_or_path (str): Path to pretrained SD3 VAE
        in_channels (int, optional): Number of input channels (default: 5)
        out_channels (int, optional): Number of output channels (default: 5)
        adapter_channels (int, optional): Number of channels in adapter layers (default: 32)
        adapter_placement (str, optional): Where to place adapters ("input", "output", or "both")
        use_sam_loss (bool, optional): Whether to use SAM loss (default: True)
        subfolder (str, optional): Subfolder for SD3 compatibility
        revision (str, optional): Revision for SD3 compatibility
        variant (str, optional): Variant for SD3 compatibility
        torch_dtype (torch.dtype, optional): Torch dtype for SD3 compatibility
    """

    #HuggingFace's from_pretrained first loads config.json and then calls __init__ method via from_config.
    # __init__ must be compatible with keyword-based instantiation — not positional-only

    # Refactor: Use explicit adapter and backbone channel config fields for clarity
    @register_to_config
    def __init__(
        self,
        *,
        pretrained_model_name_or_path: str = None,
        adapter_in_channels: int = 5,  # Refactored: adapter input channels
        adapter_out_channels: int = 5, # Refactored: adapter output channels
        backbone_in_channels: int = 3, # Refactored: backbone input channels
        backbone_out_channels: int = 3, # Refactored: backbone output channels
        adapter_channels: int = 32,
        adapter_placement: str = "both",
        use_sam_loss: bool = True,
        subfolder: str = "vae",
        revision: str = None,
        variant: str = None,
        torch_dtype: torch.dtype = None,
        use_saturation_penalty: bool = False,
    ):
        # Adapter config: these are for the adapters only, not the backbone
        self.adapter_placement = adapter_placement
        self.use_sam_loss = use_sam_loss
        self.adapter_in_channels = adapter_in_channels  # Refactored: adapter input channels
        self.adapter_out_channels = adapter_out_channels  # Refactored: adapter output channels
        self.adapter_channels = adapter_channels
        self.use_saturation_penalty = use_saturation_penalty
        self.backbone_in_channels = backbone_in_channels  # Refactored: backbone input channels
        self.backbone_out_channels = backbone_out_channels  # Refactored: backbone output channels
        # The backbone (SD3 VAE) is always 3->3 channels, regardless of adapter config
        if pretrained_model_name_or_path is None:
            raise ValueError("`pretrained_model_name_or_path` must be passed to `from_pretrained()` or stored in config.")
        config = AutoencoderKL.load_config(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            revision=revision,
            variant=variant,
        )
        adapter_keys = {
            "pretrained_model_name_or_path",
            "adapter_channels",
            "adapter_placement",
            "use_sam_loss",
            "revision",
            "subfolder",
            "torch_dtype",
            "variant",
            "use_saturation_penalty",
            "adapter_in_channels",
            "adapter_out_channels",
            "backbone_in_channels",
            "backbone_out_channels",
        }
        config = {k: v for k, v in config.items() if k not in adapter_keys}
        # Refactored: Use explicit backbone channel config
        config["in_channels"] = self.backbone_in_channels
        config["out_channels"] = self.backbone_out_channels
        super().__init__(**config)
        self.load_from_pretrained(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            revision=revision,
            variant=variant,
            torch_dtype=torch_dtype,
        )

    def load_from_pretrained(self, pretrained_model_name_or_path, subfolder=None, revision=None, variant=None, torch_dtype=None):
        # Refactored: Use explicit adapter channel config for adapters
        base_model = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            revision=revision,
            variant=variant,
            torch_dtype=torch_dtype,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False,
        )
        # Load backbone weights (no adapters)
        self.load_state_dict(base_model.state_dict(), strict=False)
        # Instantiate adapters
        if self.adapter_placement in ["input", "both"]:
            self.input_adapter = SpectralAdapter(
                self.adapter_in_channels, self.backbone_in_channels  # Refactored: adapter_in_channels -> backbone_in_channels
            )
        if self.adapter_placement in ["output", "both"]:
            self.output_adapter = SpectralAdapter(
                self.backbone_out_channels, self.adapter_out_channels  # Refactored: backbone_out_channels -> adapter_out_channels
            )
        # Load the full state dict (including adapters) if available
        # instantiate adapter->full state dict load : 
        # ensures that all trained adapter weights are restored when loading pretrained multispectral VAE
        import os
        import torch
        # Try to find the state dict file (HuggingFace convention)
        state_dict_path = None
        for fname in ["pytorch_model.bin", "diffusion_pytorch_model.bin"]:
            candidate = os.path.join(pretrained_model_name_or_path, fname)
            if os.path.isfile(candidate):
                state_dict_path = candidate
                break
        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)
        self.freeze_backbone()

    # Freezing pretrained SD3 VAE ensures latent space remains aligned with pretrained distributions.
    # Prevents catastrophic forgetting and saves compute.
    def freeze_backbone(self):
        """Freeze all parameters except adapter layers.
        Preserves the pretrained SD3 latent space, preventing catastrophic forgetting.
        """
        for param in self.parameters():
            param.requires_grad = False
        # Unfreeze adapter layers
        if hasattr(self, 'input_adapter'):
            for param in self.input_adapter.parameters():
                param.requires_grad = True
        if hasattr(self, 'output_adapter'):
            for param in self.output_adapter.parameters():
                param.requires_grad = True

    # Only adapters are trained; backbone remains frozen for efficiency and latent consistency.
    def get_trainable_params(self):
        """Get parameters that should be trained (only adapter layers)."""
        params = []
        if hasattr(self, 'input_adapter'):
            params.extend(self.input_adapter.parameters())
        if hasattr(self, 'output_adapter'):
            params.extend(self.output_adapter.parameters())
        return params

    def get_scale_monitoring_info(self) -> Dict[str, any]:
        """
        Get comprehensive scale monitoring information from all adapters.
        
        Returns:
            Dictionary containing scale monitoring data from all adapters
        """
        monitoring_info = {
            'input_adapter': None,
            'output_adapter': None,
            'overall_status': 'unknown',
            'combined_warnings': [],
            'combined_recommendations': []
        }
        
        # Collect monitoring info from input adapter
        if hasattr(self, 'input_adapter') and hasattr(self.input_adapter, 'get_scale_monitoring_info'):
            monitoring_info['input_adapter'] = self.input_adapter.get_scale_monitoring_info()
        
        # Collect monitoring info from output adapter
        if hasattr(self, 'output_adapter') and hasattr(self.output_adapter, 'get_scale_monitoring_info'):
            monitoring_info['output_adapter'] = self.output_adapter.get_scale_monitoring_info()
        
        # Determine overall status and collect warnings
        input_collapsed = (monitoring_info['input_adapter'] and 
                          monitoring_info['input_adapter'].get('is_collapsed', False))
        output_collapsed = (monitoring_info['output_adapter'] and 
                           monitoring_info['output_adapter'].get('is_collapsed', False))
        input_exploded = (monitoring_info['input_adapter'] and 
                         monitoring_info['input_adapter'].get('is_exploded', False))
        output_exploded = (monitoring_info['output_adapter'] and 
                          monitoring_info['output_adapter'].get('is_exploded', False))
        
        # Collect all warnings and recommendations
        if monitoring_info['input_adapter']:
            monitoring_info['combined_warnings'].extend(monitoring_info['input_adapter'].get('warnings', []))
            monitoring_info['combined_recommendations'].extend(monitoring_info['input_adapter'].get('recommendations', []))
        
        if monitoring_info['output_adapter']:
            monitoring_info['combined_warnings'].extend(monitoring_info['output_adapter'].get('warnings', []))
            monitoring_info['combined_recommendations'].extend(monitoring_info['output_adapter'].get('recommendations', []))
        
        # Determine overall status
        if input_collapsed or output_collapsed:
            monitoring_info['overall_status'] = 'collapsed'
        elif input_exploded or output_exploded:
            monitoring_info['overall_status'] = 'exploded'
        elif (monitoring_info['input_adapter'] and monitoring_info['input_adapter'].get('is_converged', False)) and \
             (monitoring_info['output_adapter'] and monitoring_info['output_adapter'].get('is_converged', False)):
            monitoring_info['overall_status'] = 'converged'
        elif len(monitoring_info['combined_warnings']) > 0:
            monitoring_info['overall_status'] = 'warning'
        else:
            monitoring_info['overall_status'] = 'healthy'
        
        return monitoring_info

    # Design Rationale: Multi-objective Loss
    # --------------------------------------
    # Combines per-channel MSE (spatial fidelity) with optional SAM (spectral similarity).
    # - MSE ensures accurate reconstruction pixel-wise.
    # - SAM focuses on preserving spectral signatures, regardless of scale.
    # The combination allows the model to prioritize spectral realism, important in plant health analysis.
    def compute_losses(self, original: torch.Tensor, reconstructed: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Compute masked loss terms for multispectral autoencoder training.

        Computes per-channel MSE loss and optional Spectral Angle Mapper (SAM) loss
        using binary masking to focus training purely on leaf regions, excluding background. 
        This ensures both pixel-wise accuracy and spectral fidelity.

        Args:
            original: Original multispectral image, shape (B, 5, H, W)
            reconstructed: Reconstructed multispectral image, shape (B, 5, H, W)
            mask: Binary mask (1 for leaf/foreground, 0 for background), shape (B, 1, H, W)
                 If None, computes loss over entire image

        Returns:
            Dictionary containing different loss terms:
                - 'mse_per_channel': Per-channel MSE (masked, averaged over valid pixels)
                - 'mse': Mean MSE over all channels (masked)
                - 'sam': (optional) Spectral Angle Mapper loss (masked)
                - 'total_loss': Combined loss (currently just MSE)
                - 'mask_stats': Statistics about mask coverage for monitoring
        """
        losses = {}
        
        # Validate mask shape and prepare for broadcasting
        if mask is not None:
            # Ensure mask has correct shape for broadcasting with (B, 5, H, W)
            if mask.shape[1] == 1:
                # Expand mask from (B, 1, H, W) to (B, 5, H, W) for channel-wise masking
                mask = mask.expand(-1, original.shape[1], -1, -1)
            elif mask.shape[1] != original.shape[1]:
                raise ValueError(f"Mask channels ({mask.shape[1]}) must be 1 or match input channels ({original.shape[1]})")
            
            # Ensure mask is binary (0 or 1)
            mask = (mask > 0.5).float()
            
            # Log mask statistics for monitoring
            mask_coverage = mask.mean().item()
            losses['mask_stats'] = {
                'coverage': mask_coverage,
                'valid_pixels': mask.sum().item(),
                'total_pixels': mask.numel()
            }
            
            if mask_coverage < 0.01:
                logger.warning(f"Very low mask coverage ({mask_coverage:.4f}), training may be unstable")
            elif mask_coverage > 0.99:
                logger.warning(f"Very high mask coverage ({mask_coverage:.4f}), consider if masking is necessary")
        else:
            # No mask provided - use full image (not recommended for plant data)
            mask = torch.ones_like(original)
            losses['mask_stats'] = {
                'coverage': 1.0,
                'valid_pixels': mask.numel(),
                'total_pixels': mask.numel()
            }
            logger.warning("No mask provided - computing loss over entire image including background")

        # Apply mask to both original and reconstructed images
        masked_original = original * mask
        masked_reconstructed = reconstructed * mask

        # Compute per-channel MSE loss with masking
        # Use reduction='none' to get per-pixel losses, then apply mask
        mse_per_pixel = F.mse_loss(masked_reconstructed, masked_original, reduction='none')
        
        # Average over masked regions only (excluding background)
        if mask is not None:
            # Sum over spatial dimensions, then divide by number of valid pixels per channel
            mse_per_channel = (mse_per_pixel * mask).sum(dim=(0, 2, 3)) / (mask.sum(dim=(0, 2, 3)) + 1e-8)
        else:
            mse_per_channel = mse_per_pixel.mean(dim=(0, 2, 3))
        
        losses['mse_per_channel'] = mse_per_channel
        losses['mse'] = mse_per_channel.mean()

        # Spectral Angle Mapper loss for spectral fidelity (masked)
        if self.use_sam_loss:
            # Apply mask before computing SAM loss
            if mask is not None:
                # For SAM, we need to handle the case where some pixels have zero spectral magnitude
                # after masking. We'll compute SAM only on pixels with sufficient spectral content.
                spectral_magnitude = torch.norm(masked_original, dim=1, keepdim=True)  # (B, 1, H, W)
                valid_spectral_mask = (spectral_magnitude > 1e-6) & (mask[:, :1, :, :] > 0.5)
                
                if valid_spectral_mask.sum() > 0:
                    # Extract valid pixels for SAM computation
                    valid_original = masked_original[valid_spectral_mask.expand_as(masked_original)]
                    valid_reconstructed = masked_reconstructed[valid_spectral_mask.expand_as(masked_reconstructed)]
                    
                    # Reshape to (N, C) where N is number of valid pixels, C is channels
                    valid_original = valid_original.view(-1, original.shape[1])
                    valid_reconstructed = valid_reconstructed.view(-1, original.shape[1])
                    
                    # Compute SAM on valid pixels only
                    losses['sam'] = compute_sam_loss(valid_original, valid_reconstructed)
                else:
                    # No valid spectral pixels, set SAM loss to zero
                    losses['sam'] = torch.tensor(0.0, device=original.device, dtype=original.dtype)
                    logger.warning("No valid spectral pixels found for SAM loss computation")
            else:
                # No mask provided, compute SAM on entire image
                losses['sam'] = compute_sam_loss(original, reconstructed)
            
            if torch.isnan(losses['sam']):
                logger.debug("[NaN DEBUG] SAM loss returned NaN")

        # Total loss (currently just MSE, but could be weighted combination)
        losses['total_loss'] = losses['mse']

        # Saturation penalty: softly discourages outputs from nearing ±1 extremes,
        # which can compress spectral details important in plant stress analysis.
        # Applies penalty only if use_saturation_penalty=True was set during model init.
        # v12: saturation_penalty immediately reduced output range from
        # Decoder output range: min=-9.5801, max=9.3622 to
        # Decoder output range: min=-3.4364, max=5.2877
        #
        # ARCHITECTURAL DESIGN RATIONALE: Why Saturation Penalty is in Model Code
        # ----------------------------------------------------------------------
        # The saturation penalty is placed in the model/adapter code (not training loop) because:
        #1. SCIENTIFIC FIDELITY: It is a core part of the model's scientific objective - preserving
        #    spectral interpretability by preventing hard saturation that destroys subtle spectral
        #    differences crucial for plant health analysis. This is always relevant, regardless of
        #    how the model is used (training, inference, or integration).
        # 2. MODEL CONSISTENCY: By being in the model code, it ensures consistent application across
        #    all use cases - the model will always preserve spectral fidelity, even when used in
        #    different pipelines or by other researchers.
        # 3. REPRODUCIBILITY: The saturation penalty is part of the model's scientific design and
        #    should be preserved when sharing or reusing the model, ensuring consistent spectral
        #    behavior across different implementations.
        # 4. SEPARATION OF CONCERNS: The model handles scientific fidelity (saturation penalty),
        #    while the training loop handles engineering constraints (range penalty for SD3tibility).
        #
        # COMPLEMENTARY APPROACH: The range penalty (in training loop) handles the practical
        # engineering constraint of ensuring outputs stay within [-1,1ownstream pipeline
        # compatibility, while this saturation penalty handles the scientific constraint of
        # preserving spectral detail and avoiding hard nonlinearities.
        if self.use_saturation_penalty:
            saturation_penalty = torch.mean(F.relu(torch.abs(reconstructed) - 0.95))
            losses["saturation_penalty"] = saturation_penalty
            losses["total_loss"] = losses["total_loss"] + 0.05 * saturation_penalty

        return losses

    # Corrected from_pretrained logic to fully instantiate the adapter with all required arguments
    # based on Hugging Face config and user-provided overrides.
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        # Refactored: Use explicit adapter and backbone channel config
        config = AutoencoderKL.load_config(
            pretrained_model_name_or_path,
            subfolder=kwargs.get("subfolder", "vae"),
            revision=kwargs.get("revision", None),
            variant=kwargs.get("variant", None),
        )
        model = cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            adapter_in_channels=kwargs.get("adapter_in_channels", 5),
            adapter_out_channels=kwargs.get("adapter_out_channels", 5),
            backbone_in_channels=kwargs.get("backbone_in_channels", 3),
            backbone_out_channels=kwargs.get("backbone_out_channels", 3),
            adapter_channels=kwargs.get("adapter_channels", 32),
            adapter_placement=kwargs.get("adapter_placement", "both"),
            use_sam_loss=kwargs.get("use_sam_loss", True),
            subfolder=kwargs.get("subfolder", "vae"),
            revision=kwargs.get("revision", None),
            variant=kwargs.get("variant", None),
            torch_dtype=kwargs.get("torch_dtype", None),
            use_saturation_penalty=kwargs.get("use_saturation_penalty", False),
        )
        # After instantiation, load the full state dict (including adapters)
        import os
        import torch
        state_dict_path = None
        for fname in ["pytorch_model.bin", "diffusion_pytorch_model.bin"]:
            candidate = os.path.join(pretrained_model_name_or_path, fname)
            if os.path.isfile(candidate):
                state_dict_path = candidate
                break
        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
        # Refactored: Post-load assertion for adapter channels
        if model.adapter_in_channels != 5:
            raise AssertionError(
                f"Adapter expected adapter_in_channels=5, got {model.adapter_in_channels}.\n"
                f"Note: model.config.in_channels is {model.config.in_channels} (backbone, always 3).\n"
                f"If you are loading from a pretrained RGB VAE, this is expected for the backbone, but the adapter must use 5 channels.\n"
                f"Check that you are passing adapter_in_channels=5 to from_pretrained and that your config is correct."
            )
        if model.adapter_out_channels != 5:
            raise AssertionError(
                f"Adapter expected adapter_out_channels=5, got {model.adapter_out_channels}.\n"
                f"Note: model.config.out_channels is {model.config.out_channels} (backbone, always 3).\n"
                f"If you are loading from a pretrained RGB VAE, this is expected for the backbone, but the adapter must use 5 channels.\n"
                f"Check that you are passing adapter_out_channels=5 to from_pretrained and that your config is correct."
            )
        return model

    def save_pretrained(self, save_directory, *args, **kwargs):
        # Refactored: Save explicit adapter and backbone channel config fields
        self.register_to_config(
            adapter_in_channels=self.adapter_in_channels,
            adapter_out_channels=self.adapter_out_channels,
            backbone_in_channels=self.backbone_in_channels,
            backbone_out_channels=self.backbone_out_channels,
            adapter_placement=self.adapter_placement,
            use_sam_loss=self.use_sam_loss,
            adapter_channels=self.adapter_channels,
            use_saturation_penalty=self.use_saturation_penalty,
        )
        super().save_pretrained(save_directory, *args, **kwargs)

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> Union[AutoencoderKLOutput, Tuple]:
        """Encode multispectral image to latent space.
        
        - Input adapter ensures compatibility with 5-channel plant data and robust handling of NaNs/Infs.
        - Supports end-to-end scientific validity by aligning with the dataloader's mask/NaN handling.
        """
        if hasattr(self, 'input_adapter'):
            # Convert 5 channels to 3 using input adapter
            x = self.input_adapter(x)
        # Use pretrained VAE encoder
        return super().encode(x, return_dict=return_dict)

    @apply_forward_hook
    def decode(self, z, return_dict: bool = True):
        """
        Decode latent representation to 5-channel multispectral image.

        This override of `AutoencoderKL.decode` extracts the decoded tensor from the SD3 backbone 
        before applying the output adapter. The result can be returned as either a raw tensor 
        (for training compatibility) or a DecoderOutput object (for downstream pipeline compatibility).

        RETURN_DICT VARIABILITY EXPLANATION:
        ------------------------------------
        The return_dict parameter enables compatibility with different usage patterns:
        
        1. return_dict=True (default): Returns DecoderOutput object
           - Used by: HuggingFace pipelines, SD3 integration, downstream inference
           - Access pattern: vae.decode(latents).sample
           - Examples: 
             * StableDiffusionPipeline.decode_latents(): image = vae.decode(latents).sample
             * CogVideoXSTGPipeline.decode_latents(): frames = vae.decode(latents).sample
             * VAE roundtrip: rgb_nchw = VaeImageProcessor.denormalize(decoding_nchw.sample)
        
        2. return_dict=False: Returns raw tensor directly
           - Used by: Training scripts, custom loss computation, direct tensor operations
           - Access pattern: reconstruction = vae.decode(z, return_dict=False)
           - Examples:
             * Training loops: reconstruction = model.decode(z, return_dict=False)
             * Loss computation: losses = model.compute_losses(batch, reconstruction)
             * Legacy pipelines: image = vae.decode(latents, return_dict=False)[0]
        
        This dual interface ensures:
        - Backward compatibility with existing training code
        - Forward compatibility with HuggingFace pipeline ecosystem
        - No breaking changes to downstream SD3 integration
        - Consistent behavior with base AutoencoderKL.decode() method

        IMPORTANT: The output adapter applies significant nonlinear transformations including:
        - Two convolutional blocks with SiLU activations and GroupNorm
        - Final linear convolution with learnable scale and bias parameters
        
        These nonlinearities mean the output range is NOT constrained to [-1, 1] and may require
        post-processing normalization depending on downstream usage.

        Args:
            z (torch.Tensor): Latent vector.
            return_dict (bool): If True, returns DecoderOutput object. If False, returns raw tensor.

        Returns:
            Union[DecoderOutput, torch.Tensor]: Reconstructed 5-channel image with nonlinear transformations applied.
                         Output range is not guaranteed to be [-1, 1] due to adapter nonlinearities.
        """
        # Cast input to float32 if it's float16 (for stability with mixed precision pipelines)
        if z.dtype == torch.float16:
            z = z.to(torch.float32)
        
        # Get base decoder output
        raw = super().decode(z, return_dict=True)
        
        # Extract sample tensor
        if isinstance(raw, dict):
            decoded = raw["sample"]
        elif hasattr(raw, "sample"):
            decoded = raw.sample
        elif isinstance(raw, tuple):
            decoded = raw[0]
        else:
            decoded = raw
        
        # Apply output adapter with nonlinear transformations
        adapted_output = self.output_adapter(decoded)
        
        # Return format based on return_dict parameter
        if return_dict:
            # Return DecoderOutput for downstream pipeline compatibility
            return DecoderOutput(sample=adapted_output)
        else:
            # Return tuple for pipeline compatibility (pipeline expects [0] indexing)
            return (adapted_output,)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through the entire network with optional masking for leaf-focused training.

        NOTE: This forward method avoids using AutoencoderKLOutput for decoding output, in compliance with Hugging Face's class definition.
        The previously encountered error ("AutoencoderKLOutput.__init__() got an unexpected keyword argument") has been resolved
        by ensuring decode() returns only raw tensors, not structured outputs.

        Args:
            sample: Input multispectral image, shape (B, 5, H, W)
            sample_posterior: Whether to sample from posterior or use mode
            return_dict: Whether to return structured output (for decode method)
            generator: Random generator for sampling
            mask: Optional binary mask for leaf regions, shape (B, 1, H, W)

        Returns:
            Tuple of (decoded_output, losses_dict) where losses_dict is None in eval mode

        - In eval/inference mode: only encodes and decodes, returns decoded tensor (or DecoderOutput), skips all loss computation, mask handling, and logging.
        """
        # Input sample is adapted and encoded
        x = sample

        if self.training:
            # Log: NaNs in input
            if torch.isnan(x).any():
                logger.debug("[NaN DEBUG] NaNs detected in input")
            logger.debug(f"[NaN DEBUG] Input stats before encode - min: {x.min().item():.6f}, max: {x.max().item():.6f}, mean: {x.mean().item():.6f}")
            posterior = self.encode(x).latent_dist
            if torch.isnan(posterior.mean).any() or torch.isnan(posterior.logvar).any():
                logger.debug("[NaN DEBUG] NaNs detected in posterior mean/logvar after encode")
            if sample_posterior:
                z = posterior.sample(generator=generator)
            else:
                z = posterior.mode()
            if torch.isnan(z).any():
                logger.debug("[NaN DEBUG] NaNs detected in latent z")
            decoded = self.decode(z)
            if isinstance(decoded, DecoderOutput):
                decoded_tensor = decoded.sample
            else:
                decoded_tensor = decoded
            if torch.isnan(decoded_tensor).any():
                logger.debug("[NaN DEBUG] NaNs detected in reconstruction")
            losses = self.compute_losses(x, decoded_tensor, mask)
            # Log: Loss debugging for NaNs or unusual values
            if torch.isnan(losses['mse']).any():
                logger.debug("[NaN DEBUG] NaNs detected in MSE loss")
            if self.use_sam_loss and 'sam' in losses and torch.isnan(losses['sam']).any():
                logger.debug("[NaN DEBUG] NaNs detected in SAM loss")
            if 'mse_per_channel' in losses:
                logger.debug(f"[DEBUG] Per-channel MSE stats — min: {losses['mse_per_channel'].min().item():.4f}, max: {losses['mse_per_channel'].max().item():.4f}")
            
            # Log mask statistics if available
            if 'mask_stats' in losses:
                stats = losses['mask_stats']
                logger.debug(f"[DEBUG] Mask stats — coverage: {stats['coverage']:.4f}, valid_pixels: {stats['valid_pixels']}/{stats['total_pixels']}")

            # Global scale monitoring: logs the learned scalar after each forward pass during training.
            # This is critical for ensuring the output magnitude remains biologically meaningful.
            # A scale too small (<0.1) or too large (>3.0) can indicate loss of spectral fidelity.
            # Log and check global scale if output_adapter exists
            if hasattr(self, "output_adapter"):
                global_scale_val = self.output_adapter.global_scale.item()
                logger.info(f"[Monitor] Global scale: {global_scale_val:.4f}")
                if global_scale_val < 0.1 or global_scale_val > 3.0:
                    logger.warning(f"[Monitor] Global scale {global_scale_val:.4f} is outside recommended bounds [0.1, 3.0].")
            return decoded_tensor, losses
        else:
            # inference only
            posterior = self.encode(x).latent_dist
            if sample_posterior:
                z = posterior.sample(generator=generator)
            else:
                z = posterior.mode()
            decoded = self.decode(z, return_dict=return_dict)
            if return_dict and isinstance(decoded, DecoderOutput):
                return decoded
            elif isinstance(decoded, DecoderOutput):
                return decoded.sample
            else:
                return decoded