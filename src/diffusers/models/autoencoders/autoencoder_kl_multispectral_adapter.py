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
   - Spectral attention mechanism for interpretable band selection (nonlinear)
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

   b) SpectralAttention:
      - 1×1 convolution: Learn band importance weights (linear transformation)
      - Sigmoid activation: Ensure interpretable [0,1] importance scores (nonlinear activation)
      - Element-wise multiplication: Apply learned weights to input (nonlinear due to sigmoid weights)
      - Wavelength mapping: Enable scientific visualization of band contributions

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
   - get_band_importance(): Generate interpretable visualizations
   - Per-band loss tracking for spectral fidelity analysis
   - Support for spectral signature preservation studies

Implementation Details:
---------------------
1. Model Components:
   - Pretrained SD3 VAE backbone
   - Input/output adapter layers
   - Spectral attention mechanism
   - Loss computation pipeline

2. Training Integration:
   - Parameter isolation for efficient fine-tuning
   - Loss term balancing
   - Spectral fidelity preservation
   - Band importance tracking

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
   - Develop novel spectral attention mechanisms
   - Investigate band correlation patterns
   - Study spectral signature preservation
   - Explore adaptive normalization strategies
   - Design spectral-aware loss functions

2. Model Architecture:
   - Propose new adapter architectures
   - Develop spectral correlation models
   - Create band importance metrics
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

Usage:
    # Initialize with pretrained SD3 VAE
    vae = AutoencoderKLMultispectralAdapter.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        adapter_placement="both",  # or "input" or "output"
        use_spectral_attention=True,
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
# After encountering multiple initialization and argument errors,
# we determined that Hugging Face's `from_pretrained()` logic uses
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

# Set up logger for this module
logger = logging.getLogger(__name__)
from ...configuration_utils import ConfigMixin, register_to_config
from ...utils.accelerate_utils import apply_forward_hook
from ..modeling_outputs import AutoencoderKLOutput
from .autoencoder_kl import AutoencoderKL
from .vae import DecoderOutput

#
# Design Rationale: Spectral Attention
# ------------------------------------
# Applies learnable per-band weights via 1×1 conv + sigmoid to modulate the importance of each band.
# This aids interpretability and allows the model to emphasize diagnostically relevant wavelengths.
# The weights can later be visualized and mapped to biological wavelengths for scientific insight.
#
class SpectralAttention(nn.Module):
    """Attention mechanism for spectral band selection.

    This module learns to weight the importance of each spectral band
    during the adaptation process. It helps the model focus on the most
    relevant bands for the task while maintaining spectral relationships.
    
    IMPORTANT: This module applies nonlinear transformations:
    - 1×1 convolution (linear)
    - Sigmoid activation (nonlinear: maps to [0,1] range)
    - Element-wise multiplication with input (nonlinear due to sigmoid weights)
    
    - Enables explainable AI attention weights can be visualized and mapped to wavelengths for scientific interpretability.
    - Supports plant science by highlighting diagnostically relevant bands.
    - Nonlinear attention mechanism allows for complex band interaction modeling.
    """

    def __init__(self, num_bands: int):
        super().__init__()
        # Simple 1x1 convolution followed by sigmoid to learn band weights
        self.attention = nn.Sequential(
            nn.Conv2d(num_bands, num_bands, kernel_size=1),  # Linear transformation
            nn.Sigmoid()  # Nonlinear activation: ensures weights are between 0 and 1
        )

        # Store wavelength information for interpretability
        # These wavelengths correspond to specific bands in the hyperspectral data
        self.wavelengths = {
            0: 474.73,  # Band 9: Blue - chlorophyll absorption
            1: 538.71,  # Band 18: Green - healthy vegetation
            2: 650.665, # Band 32: Red - chlorophyll content
            3: 730.635, # Band 42: Red-edge - stress detection
            4: 850.59   # Band 55: NIR - leaf health
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
            # Create a dummy input to get attention weights
            # Using ones ensures we get the base attention values
            dummy_input = torch.ones(1, len(self.wavelengths), 1, 1)
            attention_weights = self.attention(dummy_input).squeeze()

            # Map weights to wavelengths for interpretability
            return {self.wavelengths[i]: float(weight)
                   for i, weight in enumerate(attention_weights)}

#
# Design Rationale: SpectralAdapter
# ---------------------------------
# These adapter layers transform between 5-channel multispectral inputs and the 3-channel
# format expected by the SD3 VAE. The input adapter maps 5→3 and output adapter maps 3→5.
# We use:
# - 3×3 convs for efficient spatial-spectral processing
# - GroupNorm for stability with small batch sizes (nonlinear normalization)
# - SiLU (Swish) activation for smoother gradients compared to ReLU (nonlinear activation)
# - Spectral attention with sigmoid for band weighting (nonlinear attention)
# The adapters can be placed at input/output/both to allow ablation studies and flexibility.
# 
# IMPORTANT: The nonlinear transformations (SiLU, GroupNorm, sigmoid attention) mean that
# the output range is not constrained to [-1, 1] and may require post-processing normalization.
#
class SpectralAdapter(nn.Module):
    """Adapter module for converting between 3 and 5 spectral channels.

    This module handles the conversion between the 5-channel multispectral
    input and the 3-channel RGB-like format expected by the SD3 VAE.
    It includes spectral attention and a series of convolutions to learn
    the optimal transformation while preserving spectral information.
    
    IMPORTANT: This adapter applies significant nonlinear transformations:
    - Spectral attention with sigmoid activation and element-wise multiplication
    - Two convolutional blocks with SiLU activations and GroupNorm
    - Final linear convolution without activation
    
    These nonlinearities mean the output range is NOT constrained and may require
    post-processing normalization depending on downstream usage.
    
    Scientific Rationale:
    --------------------
    - Bridges the gap between 5-channel plant data and 3-channel SD3 backbone (core methodological innovation).
    - NaN-masking is essential for preventing background artifacts from propagating through the network.
    - Nonlinear transformations enable complex spectral relationships to be learned.

    """
    # NOTE: We assume that background pixels in padded multispectral input are encoded as NaN.
    # This adapter explicitly masks (zeroes out) these background pixels to avoid propagating NaNs.

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = True,
        num_bands: int = 5
    ):
        super().__init__()
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

        # Group normalization for better training stability (nonlinear normalization)
        self.norm1 = nn.GroupNorm(8, 32)
        self.norm2 = nn.GroupNorm(8, 32)

        # SiLU activation (also known as Swish) - nonlinear activation function
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply background mask: Set any NaNs to 0.0 (e.g., padding area)
        # This prevents downstream convolutional layers from propagating invalid values
        if torch.isnan(x).any():
            logger.debug("[NaN DEBUG] NaNs found in adapter input, replacing with 0.0 (masked background).")
        x = torch.nan_to_num(x, nan=0.0)

        # Log adapter input stats for NaN debugging
        logger.debug(f"[NaN DEBUG] SpectralAdapter input stats - min: {x.min().item():.6f}, max: {x.max().item():.6f}, mean: {x.mean().item():.6f}")

        # Apply spectral attention if enabled (nonlinear: sigmoid + element-wise multiplication)
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

        # Log adapter output stats for NaN debugging
        if torch.isnan(x).any():
            logger.debug("[NaN DEBUG] NaNs in SpectralAdapter output.")
        logger.debug(f"[NaN DEBUG] SpectralAdapter output stats - min: {x.min().item():.6f}, max: {x.max().item():.6f}, mean: {x.mean().item():.6f}")
        return x

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

        # Optionally clamp to prevent propagation of small/large values
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
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
# - use_spectral_attention: Enables band-wise weighting for interpretable spectral relevance.
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
    5. Including spectral attention and specialized losses

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

    @register_to_config
    def __init__(
        self,
        *, # forces all arguments to be keyword-only, which is expected by the Diffusers .from_pretrained() logic.
        #CLI command needs --base_model_path "stabilityai/stable-diffusion-3-medium-diffusers"
        pretrained_model_name_or_path: str = None,  # TODO add default
        in_channels: int = 5,
        out_channels: int = 5,
        adapter_channels: int = 32,
        adapter_placement: str = "both",
        use_spectral_attention: bool = True,
        use_sam_loss: bool = True,
        subfolder: str = "vae",
        revision: str = None,
        variant: str = None,
        torch_dtype: torch.dtype = None,
    ):
        # Set config attributes on self
        self.adapter_placement = adapter_placement
        self.use_spectral_attention = use_spectral_attention
        self.use_sam_loss = use_sam_loss
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.adapter_channels = adapter_channels
        
        # Ensure in_channels and out_channels are set to 5 for multispectral data
        if in_channels != 5:
            logger.warning(f"in_channels should be 5 for multispectral data, got {in_channels}")
        if out_channels != 5:
            logger.warning(f"out_channels should be 5 for multispectral data, got {out_channels}")

        # Check for required pretrained_model_name_or_path
        if pretrained_model_name_or_path is None:
            raise ValueError("`pretrained_model_name_or_path` must be passed to `from_pretrained()` or stored in config.")
        # Load the base config and pass it to the parent __init__.
        config = AutoencoderKL.load_config(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            revision=revision,
            variant=variant,
        )
        # Remove adapter-specific config keys before AutoencoderKL.__init__
        # otherwise loading from pretrained config.json has base class encounter unexpected keywords 
        adapter_keys = {
            "pretrained_model_name_or_path",
            "adapter_channels",
            "adapter_placement",
            "use_spectral_attention",
            "use_sam_loss",
            "in_channels",
            "out_channels",
            "revision",
            "subfolder",
            "torch_dtype",
            "variant",
        }
        config = {k: v for k, v in config.items() if k not in adapter_keys}
        super().__init__(**config)
        
        # Override the config values with our multispectral settings
        # This ensures that in_channels and out_channels are set to 5 regardless of the base model
        self.config.in_channels = in_channels
        self.config.out_channels = out_channels
        
        self.load_from_pretrained(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            revision=revision,
            variant=variant,
            torch_dtype=torch_dtype,
        )

    def load_from_pretrained(self, pretrained_model_name_or_path, subfolder=None, revision=None, variant=None, torch_dtype=None):
        # This method ensures compatibility with Hugging Face's pretrained models, while allowing for custom adapter logic.
        base_model = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            revision=revision,
            variant=variant,
            torch_dtype=torch_dtype,
        )
        self.load_state_dict(base_model.state_dict(), strict=False)

        # Use self.adapter_placement, self.use_spectral_attention, etc.
        if self.adapter_placement in ["input", "both"]:
            self.input_adapter = SpectralAdapter(
                self.in_channels, 3,
                use_attention=self.use_spectral_attention,
                num_bands=self.in_channels
            )
        if self.adapter_placement in ["output", "both"]:
            self.output_adapter = SpectralAdapter(
                3, self.out_channels,
                use_attention=self.use_spectral_attention,
                num_bands=self.out_channels
            )
        # Freeze backbone by default
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

        return losses

    # Corrected from_pretrained logic to fully instantiate the adapter with all required arguments
    # based on Hugging Face config and user-provided overrides.
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Custom `from_pretrained` to load weights into AutoencoderKLMultispectralAdapter
        from base AutoencoderKL weights (e.g., SD3 RGB VAE). This avoids issues with
        Hugging Face's internal handling of pretrained_model_name_or_path being passed twice.
        """
        # Step 1: Load base model config
        config = AutoencoderKL.load_config(
            pretrained_model_name_or_path,
            subfolder=kwargs.get("subfolder", "vae"),
            revision=kwargs.get("revision", None),
            variant=kwargs.get("variant", None),
        )

        # Step 2: Instantiate adapter model with all config + kwargs
        # Ensure in_channels and out_channels are explicitly set to 5 for multispectral data
        model = cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            in_channels=kwargs.get("in_channels", 5),  # Force 5 channels for multispectral
            out_channels=kwargs.get("out_channels", 5),  # Force 5 channels for multispectral
            adapter_channels=kwargs.get("adapter_channels", 32),
            adapter_placement=kwargs.get("adapter_placement", "both"),
            use_spectral_attention=kwargs.get("use_spectral_attention", True),
            use_sam_loss=kwargs.get("use_sam_loss", True),
            subfolder=kwargs.get("subfolder", "vae"),
            revision=kwargs.get("revision", None),
            variant=kwargs.get("variant", None),
            torch_dtype=kwargs.get("torch_dtype", None),
        )
        
        # Verify configuration was set correctly
        if model.config.in_channels != 5:
            logger.warning(f"in_channels was not set correctly: {model.config.in_channels} != 5")
        if model.config.out_channels != 5:
            logger.warning(f"out_channels was not set correctly: {model.config.out_channels} != 5")

        return model

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
    - Spectral attention with sigmoid activation and element-wise multiplication
    - Two convolutional blocks with SiLU activations and GroupNorm
    - Final linear convolution
    
    These nonlinearities mean the output range is NOT constrained to [-1, 1] and may require
    post-processing normalization depending on downstream usage.

    Args:
        z (torch.Tensor): Latent vector.
        return_dict (bool): If True, returns DecoderOutput object. If False, returns raw tensor.

    Returns:
        Union[DecoderOutput, torch.Tensor]: Reconstructed 5-channel image with nonlinear transformations applied.
                     Output range is not guaranteed to be [-1, 1] due to adapter nonlinearities.
    """
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
            # Return raw tensor for training script compatibility
            return adapted_output

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
        """
        # Input sample is adapted and encoded
        x = sample

        # Log: NaNs in input
        if torch.isnan(x).any():
            logger.debug("[NaN DEBUG] NaNs detected in input")

        # Optional: Print input stats before any adapter/encoder to diagnose anomalies in early pipeline
        logger.debug(f"[NaN DEBUG] Input stats before encode - min: {x.min().item():.6f}, max: {x.max().item():.6f}, mean: {x.mean().item():.6f}")

        posterior = self.encode(x).latent_dist

        # Log: NaNs in posterior distribution
        if torch.isnan(posterior.mean).any() or torch.isnan(posterior.logvar).any():
            logger.debug("[NaN DEBUG] NaNs detected in posterior mean/logvar after encode")

        # Choose whether to sample from posterior or use mode
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        # Log: NaNs in latent z
        if torch.isnan(z).any():
            logger.debug("[NaN DEBUG] NaNs detected in latent z")

        # Decode to obtain reconstructed output
        decoded = self.decode(z)
        # Ensure decoded is a tensor, not DecoderOutput
        if isinstance(decoded, DecoderOutput):
            decoded_tensor = decoded.sample
        else:
            decoded_tensor = decoded

        # Log: NaNs in reconstruction
        if torch.isnan(decoded_tensor).any():
            logger.debug("[NaN DEBUG] NaNs detected in reconstruction")

        # Compute training losses if in training mode
        if self.training:
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
            
            return decoded_tensor, losses

        # In evaluation mode, return decoded output and None for losses
        return decoded_tensor, None