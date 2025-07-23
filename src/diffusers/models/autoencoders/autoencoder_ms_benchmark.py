"""
Minimal Multispectral VAE Benchmark

This is a bare-bones implementation that adds only the absolutely necessary
components to process 5-channel multispectral data using the original SD3 VAE.
No sophisticated features - just basic 5->3->5  basic 1x1 convs for channel conversion.

Baseline comparison for the more sophisticated multispectral adapter.

Required Training Interface Methods
- freeze_backbone() - Freezes backbone, unfreezes adapters
- get_trainable_params() - Returns only adapter parameters
- ompute_losses() - Minimal MSE loss with mask support

Configuration Attributes
- All expected config parameters (adapter_placement, use_spectral_attention, etc.)
- Compatible from_pretrained() accepting same parameters as sophisticated version
- save_pretrained() method for checkpointing

Training Script Integration
- Updated import to use benchmark model: AutoencoderMSBenchmark as AutoencoderKLMultispectralAdapter
- forward() method supports mask parameter and returns (reconstruction, losses) in training mode

Fake Features for Compatibility
- global_scale parameter in adapters (for scale logging)
- FakeAttention class with get_band_importance() method
- Placeholder monitoring info to avoid training script errors
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ...loaders.single_file_model import FromOriginalModelMixin
from ...utils import deprecate
from ...utils.accelerate_utils import apply_forward_hook
from ..attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
    FusedAttnProcessor2_0,
)
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from .vae import Decoder, DecoderOutput, DiagonalGaussianDistribution, Encoder
from .autoencoder_kl import AutoencoderKL


class SimpleChannelAdapter(nn.Module):
    """Minimal channel adapter - just a single convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # Add fake global_scale for training script compatibility
        self.global_scale = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AutoencoderMSBenchmark(AutoencoderKL):
    """
    Minimal multispectral VAE benchmark implementation.
    
    Takes the original AutoencoderKL and adds only basic 5->3->5 channel adapters.
    No sophisticated features - just bare minimum for 5-channel processing.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 5,  # Changed from 3 to 5
        out_channels: int = 5,  # Changed from 3 to 5
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        shift_factor: Optional[float] = None,
        latents_mean: Optional[Tuple[float]] = None,
        latents_std: Optional[Tuple[float]] = None,
        force_upcast: float = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        mid_block_add_attention: bool = True,
        # Training script compatibility parameters
        adapter_placement: str = "both",
        use_spectral_attention: bool = False,  # Disabled for benchmark
        use_sam_loss: bool = False,  # Disabled for benchmark
        adapter_in_channels: int = 5,
        adapter_out_channels: int = 5,
        backbone_in_channels: int = 3,
        backbone_out_channels: int = 3,
        use_saturation_penalty: bool = False,  # Disabled for benchmark
    ):
        # Store adapter config attributes for training script compatibility
        self.adapter_placement = adapter_placement
        self.use_spectral_attention = use_spectral_attention
        self.use_sam_loss = use_sam_loss
        self.adapter_in_channels = adapter_in_channels
        self.adapter_out_channels = adapter_out_channels
        self.backbone_in_channels = backbone_in_channels
        self.backbone_out_channels = backbone_out_channels
        self.use_saturation_penalty = use_saturation_penalty
        
        # Initialize parent with 3 channels (SD3 backbone)
        super().__init__(
            in_channels=3,  # Backbone always uses 3 channels
            out_channels=3,  # Backbone always uses 3 channels
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
            latents_mean=latents_mean,
            latents_std=latents_std,
            force_upcast=force_upcast,
            use_quant_conv=use_quant_conv,
            use_post_quant_conv=use_post_quant_conv,
            mid_block_add_attention=mid_block_add_attention,
        )
        
        # Add minimal channel adapters
        self.input_adapter = SimpleChannelAdapter(5, 3)  # 5->3 for encoder
        self.output_adapter = SimpleChannelAdapter(3, 5)  # 3->5 for decoder
        
        # Add fake attention for training script compatibility if needed
        if use_spectral_attention:
            self.input_adapter.attention = FakeAttention()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load from pretrained SD3 VAE and add minimal adapters."""
        # Extract base model loading kwargs
        base_kwargs = {
            "subfolder": kwargs.get("subfolder", "vae"),
            "revision": kwargs.get("revision", None),
            "variant": kwargs.get("variant", None),
            "torch_dtype": kwargs.get("torch_dtype", None),
        }
        
        # Load the base model
        base_model = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            **base_kwargs
        )
        
        # Create benchmark model with training script compatibility parameters
        model = cls(
            adapter_placement=kwargs.get("adapter_placement", "both"),
            use_spectral_attention=kwargs.get("use_spectral_attention", False),
            use_sam_loss=kwargs.get("use_sam_loss", False),
            adapter_in_channels=kwargs.get("adapter_in_channels", 5),
            adapter_out_channels=kwargs.get("adapter_out_channels", 5),
            backbone_in_channels=kwargs.get("backbone_in_channels", 3),
            backbone_out_channels=kwargs.get("backbone_out_channels", 3),
            use_saturation_penalty=kwargs.get("use_saturation_penalty", False),
        )
        
        # Copy backbone weights
        model.load_state_dict(base_model.state_dict(), strict=False)
        
        return model

    def save_pretrained(self, save_directory, *args, **kwargs):
        """Save the model."""
        # Register config parameters for saving
        self.register_to_config(
            adapter_placement=self.adapter_placement,
            use_spectral_attention=self.use_spectral_attention,
            use_sam_loss=self.use_sam_loss,
            adapter_in_channels=self.adapter_in_channels,
            adapter_out_channels=self.adapter_out_channels,
            backbone_in_channels=self.backbone_in_channels,
            backbone_out_channels=self.backbone_out_channels,
            use_saturation_penalty=self.use_saturation_penalty,
        )
        super().save_pretrained(save_directory, *args, **kwargs)

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

    def get_trainable_params(self):
        """Get parameters that should be trained (only adapter layers)."""
        params = []
        if hasattr(self, 'input_adapter'):
            params.extend(self.input_adapter.parameters())
        if hasattr(self, 'output_adapter'):
            params.extend(self.output_adapter.parameters())
        return params

    def compute_losses(self, original: torch.Tensor, reconstructed: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Compute minimal loss for benchmark comparison (MSE only)."""
        losses = {}
        
        # Simple MSE loss
        if mask is not None:
            # Apply mask if provided
            masked_original = original * mask
            masked_reconstructed = reconstructed * mask
            mse_loss = F.mse_loss(masked_reconstructed, masked_original, reduction='none')
            # Average over masked regions only
            valid_pixels = mask.sum() + 1e-8
            losses['mse'] = (mse_loss * mask).sum() / valid_pixels
            
            # Per-channel MSE
            mse_per_channel = torch.zeros(original.shape[1], device=original.device)
            for c in range(original.shape[1]):
                channel_loss = (mse_loss[:, c:c+1] * mask).sum() / valid_pixels
                mse_per_channel[c] = channel_loss
            losses['mse_per_channel'] = mse_per_channel
            
            # Mask statistics
            losses['mask_stats'] = {
                'coverage': mask.mean().item(),
                'valid_pixels': mask.sum().item(),
                'total_pixels': mask.numel()
            }
        else:
            # No mask - compute over entire image
            losses['mse'] = F.mse_loss(reconstructed, original)
            mse_per_channel = F.mse_loss(reconstructed, original, reduction='none').mean(dim=(0, 2, 3))
            losses['mse_per_channel'] = mse_per_channel
            losses['mask_stats'] = {
                'coverage': 1.0,
                'valid_pixels': original.numel(),
                'total_pixels': original.numel()
            }
        
        losses['total_loss'] = losses['mse']
        return losses

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of 5-channel images into latents.
        """
        # Convert 5 channels to 3 using minimal adapter
        x = self.input_adapter(x)
        
        # Use original encoding logic
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)

        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(
        self, z: torch.FloatTensor, return_dict: bool = True, generator=None
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        """
        Decode latents to 5-channel images.
        """
        # Use original decoding logic to get 3-channel output
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        # Convert 3 channels to 5 using minimal adapter
        decoded = self.output_adapter(decoded)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Union[DecoderOutput, torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass with minimal 5-channel processing.
        
        If training mode and mask is provided, returns (reconstruction, losses).
        Otherwise returns reconstruction only.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        # Training mode with mask support
        if self.training and mask is not None:
            losses = self.compute_losses(x, dec, mask)
            return dec, losses
        
        # Inference mode
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec) 