"""
Multispectral VAE Adapter for Stable Diffusion 3: Core Methodological Contribution

https://huggingface.co/docs/diffusers/v0.6.0/en/api/models
https://huggingface.co/docs/diffusers/en/api/models/autoencoderkl


This module implements the central methodological contribution of the thesis: a lightweight
adapter-based multispectral autoencoder architecture built on a pretrained SD3 backbone.
The design enables efficient processing of 5-channel spectral plant imagery while maintaining
compatibility with SD3's latent space requirements.

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
   - Spectral attention mechanism for interpretable band selection
   - Dual loss function preserving both spatial and spectral fidelity

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
      - GroupNorm: Stable training with small batch sizes typical in hyperspectral data
      - SiLU activation: Gradient-friendly nonlinearity better suited than ReLU
      - Three-layer design: Progressive feature extraction and channel adaptation

   b) SpectralAttention:
      - 1×1 convolution: Learn band importance weights
      - Sigmoid activation: Ensure interpretable [0,1] importance scores
      - Wavelength mapping: Enable scientific visualization of band contributions

2. Loss Function Design:
   a) Per-channel MSE Loss:
      - Preserves spatial structure and pixel-wise accuracy
      - Enables band-specific optimization
      - Helps identify problematic spectral bands

   b) Spectral Angle Mapper (SAM) Loss:
      - Measures spectral similarity through vector angles
      - Invariant to scaling, preserving spectral signatures
      - Weighted combination: loss = α * MSE + β * SAM
      - Configurable weights for balancing spatial vs. spectral fidelity

3. Training Strategy:
   a) Parameter Efficiency:
      - freeze_backbone(): Preserve SD3's latent space properties
      - get_trainable_params(): Enable adapter-only training
      - Minimal trainable parameters (only adapter layers)

   b) Flexible Configuration:
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
    """

    def __init__(self, num_bands: int):
        super().__init__()
        # Simple 1x1 convolution followed by sigmoid to learn band weights
        self.attention = nn.Sequential(
            nn.Conv2d(num_bands, num_bands, kernel_size=1),
            nn.Sigmoid()  # Ensure weights are between 0 and 1
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
        # Compute attention weights for each band
        attention_weights = self.attention(x)
        # Apply attention weights to input
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
# - GroupNorm for stability with small batch sizes
# - SiLU (Swish) activation for smoother gradients compared to ReLU
# The adapters can be placed at input/output/both to allow ablation studies and flexibility.
#
class SpectralAdapter(nn.Module):
    """Adapter module for converting between 3 and 5 spectral channels.

    This module handles the conversion between the 5-channel multispectral
    input and the 3-channel RGB-like format expected by the SD3 VAE.
    It includes spectral attention and a series of convolutions to learn
    the optimal transformation while preserving spectral information.
    """

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
        # First two layers use 3x3 convolutions with group normalization
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # Final layer uses 1x1 convolution for channel reduction/expansion
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=1)

        # Group normalization for better training stability
        self.norm1 = nn.GroupNorm(8, 32)
        self.norm2 = nn.GroupNorm(8, 32)

        # SiLU activation (also known as Swish)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply spectral attention if enabled
        if self.use_attention and hasattr(self, 'attention'):
            x = self.attention(x)

        # First convolutional block
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        # Second convolutional block
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        # Final channel adaptation
        x = self.conv3(x)
        return x


# Numerically stable normalization for SAM loss
def safe_normalize(tensor, dim=1, eps=1e-8):
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
        logger.warning("NaNs detected in SAM angle computation.")

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
    3. Keeping backbone frozen during training
    4. Only training the adapter layers
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

        # Check for required pretrained_model_name_or_path
        if pretrained_model_name_or_path is None:
            raise ValueError("`pretrained_model_name_or_path` must be passed to `from_pretrained()` or stored in config.")
        # Load the base config and pass it to the parent __init__.
        # We unpack the config as keyword arguments because AutoencoderKL doesn't expect a single 'config' kwarg.
        config = AutoencoderKL.load_config(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            revision=revision,
            variant=variant,
        )
        super().__init__(**config)
        self.load_from_pretrained(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            revision=revision,
            variant=variant,
            torch_dtype=torch_dtype,
        )

    def load_from_pretrained(self, pretrained_model_name_or_path, subfolder=None, revision=None, variant=None, torch_dtype=None):
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
        """Freeze all parameters except adapter layers."""
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
    def compute_losses(self, original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute various loss terms for training.

        This method computes both per-channel MSE loss and Spectral Angle Mapper (SAM)
        loss to ensure both pixel-wise accuracy and spectral fidelity.

        Args:
            original: Original multispectral image
            reconstructed: Reconstructed multispectral image

        Returns:
            Dictionary containing different loss terms
        """
        losses = {}

        # Per-channel MSE loss for pixel-wise accuracy
        mse_per_channel = F.mse_loss(reconstructed, original, reduction='none')
        mse_per_channel = mse_per_channel.mean(dim=(0, 2, 3))  # Average over batch and spatial dimensions
        losses['mse_per_channel'] = mse_per_channel

        # Overall MSE loss
        losses['mse'] = mse_per_channel.mean()

        # Spectral Angle Mapper loss for spectral fidelity
        if self.use_sam_loss:
            losses['sam'] = compute_sam_loss(original, reconstructed)

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
        model = cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            in_channels=kwargs.get("in_channels", 5),
            out_channels=kwargs.get("out_channels", 5),
            adapter_channels=kwargs.get("adapter_channels", 32),
            adapter_placement=kwargs.get("adapter_placement", "both"),
            use_spectral_attention=kwargs.get("use_spectral_attention", True),
            use_sam_loss=kwargs.get("use_sam_loss", True),
            subfolder=kwargs.get("subfolder", "vae"),
            revision=kwargs.get("revision", None),
            variant=kwargs.get("variant", None),
            torch_dtype=kwargs.get("torch_dtype", None),
        )

        return model

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> Union[AutoencoderKLOutput, Tuple]:
        """Encode multispectral image to latent space."""
        if hasattr(self, 'input_adapter'):
            # Convert 5 channels to 3 using input adapter
            x = self.input_adapter(x)
        # Use pretrained VAE encoder
        return super().encode(x, return_dict=return_dict)

    @apply_forward_hook
    def decode(self, z):
        # Update: normalize handling of parent decode outputs to avoid tuple `.sample` errors
        raw = super().decode(z, return_dict=True)
        # handle various return types from parent
        if isinstance(raw, dict):
            decoded = raw["sample"]
        elif hasattr(raw, "sample"):
            decoded = raw.sample
        elif isinstance(raw, tuple):
            decoded = raw[0]
        else:
            decoded = raw
        # apply adapter to raw decoded tensor for multispectral output
        return self.output_adapter(decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through the entire network.

        NOTE: This forward method avoids using AutoencoderKLOutput for decoding output, in compliance with Hugging Face's class definition.
        The previously encountered error ("AutoencoderKLOutput.__init__() got an unexpected keyword argument") has been resolved
        by ensuring decode() returns only raw tensors, not structured outputs.
        """
        # Input sample is adapted and encoded
        x = sample
        posterior = self.encode(x).latent_dist

        # Choose whether to sample from posterior or use mode
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        # Decode to obtain reconstructed output
        decoded = self.decode(z)

        # Compute training losses if in training mode
        if self.training:
            losses = self.compute_losses(x, decoded)
            return decoded, losses

        # In evaluation mode, return decoded output and None for losses
        return decoded, None