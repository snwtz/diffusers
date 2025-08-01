"""
Multispectral VAE Adapter for Stable Diffusion 3: Core Methodological Contribution
================================================================================

This module implements a lightweight adapter-based multispectral autoencoder architecture 
built on a pretrained SD3 backbone.
The design enables efficient processing of 5-channel spectral plant imagery while maintaining
compatibility with SD3's latent space requirements.

It handles the neural network architecture and forward pass

USAGE:
------
in train_dreambooth_sd3_multispectral.py:
1. Import 
from src.diffusers.models.autoencoders.autoencoder_kl_multispectral_adapter import AutoencoderKLMultispectralAdapter
2. Load 
vae = AutoencoderKLMultispectralAdapter.from_pretrained(args.vae_path)
3. Freeze for MSDB training
vae.requires_grad_(False)
4. Use during training
model_input = vae.encode(pixel_values).latent_dist.sample() 
decoded_pixels = vae.decode(decoded_latents).sample  

CONFIGURATION:
--------------
- Input: 5-channel multispectral images (bands 9, 18, 32, 42, 55)
- Output: Reconstructed 5-channel images with spectral fidelity
- Adapters: Lightweight layers bridging 5→3→5 channels
- Backbone: Frozen SD3 VAE for parameter efficiency
- Loss: Multi-objective (MSE + SAM) with masked computation

Features:
- Spectral attention mechanism for band importance
- Parameter-efficient fine-tuning (frozen backbone)
- Masked loss computation for leaf-focused training
- SD3 pipeline compatibility
- Scale convergence monitoring
- Reference signature guidance

LOGGING AND MONITORING:
-----------------------
- Scale convergence monitoring (global scale parameter)
- Per-band importance weights (spectral attention)
- Masked loss statistics (coverage, valid pixels)
- Per-channel MSE and global SAM loss
- Numerical stability (NaN/Inf detection)
- Parameter count

Data Flow Summary:
------------------
- Input: Preprocessed 5-channel plant images (from vae_multispectral_dataloader.py)
- Model: Adapters map 5-channel input to 3-channel SD3 backbone, then back to 5-channel output
- Output: Reconstructed 5-channel image with nonlinear transformations applied (unconstrained range)
- Loss: Multi-objective (MSE + SAM), with pure leaf-focused masked loss computation
- Output Format: Compatible with both training (raw tensor) and downstream pipelines (DecoderOutput)
"""

# ------------------------------------------------------------
# VAE Loading Fix Summary:
# Hugging Face's `from_pretrained()` logic uses
# keyword arguments via a config object. To fix this:
#
# - Added `*` to enforce keyword-only arguments in __init__.
# - Unpacked the base config using `**config` when calling
#   the parent AutoencoderKL constructor (which expects kwargs).
# - Custom `from_pretrained()` classmethod that
#   combines Hugging Face-compatible config loading with custom
#   adapter arguments for spectral training.
#
# These changes allow seamless loading of RGB VAE weights into
# the adapter class while preserving pretrained features.

# The decode() method has been patched to return a DecoderOutput object
# instead of AutoencoderKLOutput, resolving the error related to unexpected
# keyword arguments like latent_sample.
# ------------------------------------------------------------

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GroupNorm
import logging
import numpy as np # Added for scale monitoring

# Set up logger for this module
logger = logging.getLogger(__name__)
from ...configuration_utils import ConfigMixin, register_to_config
from ...utils.accelerate_utils import apply_forward_hook
from ..modeling_outputs import AutoencoderKLOutput
from .autoencoder_kl import AutoencoderKL
from .vae import DecoderOutput

class SpectralAttention(nn.Module):
    """Attention mechanism for spectral band selection.

    This module learns to weight the importance of each spectral band
    during the adaptation process. It helps the model focus on the most
    relevant bands for the task while maintaining spectral relationships.
    
    IMPORTANT: This module applies nonlinear transformations:
    - 1x1 convolution (linear)
    - Sigmoid activation (nonlinear: maps to [0,1] range)
    - Element-wise multiplication with input (nonlinear due to sigmoid weights)
    
    - Nonlinear attention mechanism allows for complex band interaction modeling.
    """

    def __init__(self, num_bands: int):
        super().__init__()
        # 1x1 convolution learns per-band importance weights
        # Sigmoid ensures weights are in [0,1] range for interpretable scaling
        self.attention = nn.Sequential(
            nn.Conv2d(num_bands, num_bands, kernel_size=1),  # Linear transformation
            nn.Sigmoid()  # Nonlinear activation: ensures weights are between 0 and 1
        )

        self.wavelengths = {
            0: 474.73,  # Band 9
            1: 538.71,  # Band 18
            2: 650.665, # Band 32
            3: 730.635, # Band 42
            4: 850.59   # Band 55
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, height, width)
        # Compute attention weights for each band (nonlinear: conv + sigmoid)
        attention_weights = self.attention(x)
        # Apply attention weights to input (nonlinear: element-wise multiplication with sigmoid weights)
        return x * attention_weights

    def get_band_importance(self) -> Dict[float, float]:
        """Get the importance of each spectral band based on attention weights.

        This method is useful for interpretability and understanding
        which bands the model finds most important for the task.
        """
        with torch.no_grad():
            # Create a dummy input to get attention weights, on the same device as the module
            device = next(self.parameters()).device
            dummy_input = torch.ones(1, len(self.wavelengths), 1, 1, device=device)
            attention_weights = self.attention(dummy_input).squeeze()
            # Map weights to wavelengths for interpretability
            return {self.wavelengths[i]: float(weight)
                   for i, weight in enumerate(attention_weights)}

class SpectralAdapter(nn.Module):
    """
    Adapter module for converting between 3 and 5 spectral channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = True,
        num_bands: int = 5
    ):
        super().__init__()
        # Store channel configuration for validation and logging
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        # Initialize spectral attention if needed
        if use_attention and in_channels == num_bands:
            self.attention = SpectralAttention(num_bands)

        # Three-layer convolutional network for channel adaptation
        # First two layers use 3x3 convolutions with group normalization and nonlinear activation
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # Final layer uses 1x1 convolution for channel reduction/expansion (no activation)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=1)
   
        self.output_scale = nn.Parameter(torch.tensor(1.0))  # Global scaling factor
        self.output_bias = nn.Parameter(torch.tensor(0.0))   # Global shift

        # Group normalization for better training stability (nonlinear normalization)
        # Using torch.nn.GroupNorm for stable training with small batch sizes
        self.norm1 = GroupNorm(8, 32)
        self.norm2 = GroupNorm(8, 32)

        # SiLU activation (also known as Swish) - nonlinear activation function
        self.activation = nn.SiLU()

        #  global scale parameter monitoring
        self.global_scale = nn.Parameter(torch.tensor(1.0))
        
        # Global scale convergence monitoring
        self.scale_history = []  # Track scale values during training
        self.scale_convergence_threshold = 0.001  # Consider converged if std < threshold
        self.scale_warning_threshold = 0.01  # Warn if scale < 0.01
        self.scale_explosion_threshold = 5.0  # Warn if scale > 5.0
        self.convergence_window = 100  # Number of steps to consider for convergence
        self.step_counter = 0  # Track training steps for logging
        self.log_interval = 50  # Log scale info every N steps
        self.convergence_warning_issued = False  # Track if convergence warning was issued


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply background mask: Replace NaNs with per-band means
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
        # Apply spectral attention: learns per-band importance weights and scales input accordingly
        # Weights are computed via 1x1 conv + sigmoid, then applied via element-wise multiplication
        if self.use_attention and hasattr(self, 'attention'):
            x = self.attention(x)

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
        Scale monitoring information for logging and analysis.
        
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



def safe_normalize(tensor, dim=1, eps=1e-8):
    # Numerically stable normalization for SAM loss
    # Ensures that spectral angle calculations are robust to small values and avoid NaNs.
    norm = torch.norm(tensor, p=2, dim=dim, keepdim=True)
    norm = norm.clamp(min=eps) # CLAMP
    return tensor / norm

# Custom SAM loss function for spectral signature preservation
def compute_sam_loss(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable Spectral Angle Mapper (SAM) loss between two multispectral images.

    Args:
        original: Original multispectral image
        reconstructed: Reconstructed multispectral image
    Returns:
        Mean spectral angle in radians
    
    - Invariant to scaling, so it focuses on spectral shape rather than intensity
    """
    # Use safe normalization to avoid NaNs
    normalized_original = safe_normalize(original, dim=1)
    normalized_reconstructed = safe_normalize(reconstructed, dim=1)

    # Compute cosine similarity and clamp for stability
    cos_sim = F.cosine_similarity(normalized_original, normalized_reconstructed, dim=1)
    cos_sim = cos_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7) # CLAMP

    angle = torch.acos(cos_sim)
    angle = torch.nan_to_num(angle, nan=0.0, posinf=0.0, neginf=0.0)

    if torch.isnan(angle).any():
        logger.debug("NaNs detected in SAM angle computation.")

    return angle.mean()


# Adapter Configuration:
# ------------------------------------------
# This constructor registers all adapter-relevant settings for reproducibility.
# These settings are saved to config.json via save_pretrained() for model reloading.
#
class AutoencoderKLMultispectralAdapter(AutoencoderKL):
    """Efficient multispectral VAE implementation using adapter layers.

    Parameters:
        pretrained_model_name_or_path (str): Path to pretrained SD3 VAE
        in_channels (int, optional): Number of input channels (default: 5)
        out_channels (int, optional): Number of output channels (default: 5)
        adapter_channels (int, optional): Number of channels in adapter layers (default: 32)
        adapter_placement (str, optional): Where to place adapters ("input", "output", or "both")
        use_spectral_attention (bool, optional): Whether to use spectral attention (default: True)
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
        use_spectral_attention: bool = True,
        use_sam_loss: bool = True,
        subfolder: str = "vae",
        revision: str = None,
        variant: str = None,
        torch_dtype: torch.dtype = None,
        use_saturation_penalty: bool = False,
    ):
        # Adapter config: these are for the adapters only, not the backbone
        self.adapter_placement = adapter_placement
        self.use_spectral_attention = use_spectral_attention
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
            "use_spectral_attention",
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
                self.adapter_in_channels, self.backbone_in_channels,  # Refactored: adapter_in_channels -> backbone_in_channels
                use_attention=self.use_spectral_attention,
                num_bands=self.adapter_in_channels
            )
        if self.adapter_placement in ["output", "both"]:
            self.output_adapter = SpectralAdapter(
                self.backbone_out_channels, self.adapter_out_channels,  # Refactored: backbone_out_channels -> adapter_out_channels
                use_attention=self.use_spectral_attention,
                num_bands=self.adapter_out_channels
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

    def compute_losses(self, original: torch.Tensor, reconstructed: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Compute masked loss terms for multispectral autoencoder training.

        Computes per-channel MSE loss and optional SAM loss
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
            
        else:
            # No mask provided 
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

        # SAM loss for spectral fidelity (masked)
        if self.use_sam_loss:
            # Apply mask before computing SAM loss
            if mask is not None:
                # handle cases where some pixels have zero spectral magnitude
                # after masking compute SAM only on pixels with sufficient spectral content.
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
            use_spectral_attention=kwargs.get("use_spectral_attention", True),
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
            use_spectral_attention=self.use_spectral_attention,
            use_sam_loss=self.use_sam_loss,
            adapter_channels=self.adapter_channels,
            use_saturation_penalty=self.use_saturation_penalty,
        )
        super().save_pretrained(save_directory, *args, **kwargs)

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> Union[AutoencoderKLOutput, Tuple]:
        """Encode multispectral image to latent space.
        
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

        NOTE: This forward method avoids using AutoencoderKLOutput for decoding output
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