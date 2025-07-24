"""
Minimal Multispectral VAE Benchmark

This is a bare-bones implementation that adds only the absolutely necessary
components to process 5-channel multispectral data using the original SD3 VAE.
No sophisticated features - just basic 5->3->5  basic 1x1 convs for channel conversion.

Baseline comparison for the more sophisticated multispectral adapter.

Required Training Interface Methods
- freeze_backbone() - Freezes backbone, unfreezes adapters
- get_trainable_params() - Returns only adapter parameters
- compute_losses() - Minimal MSE loss with mask support, SAM loss, and spectral guidance

Configuration Attributes
- All expected config parameters (adapter_placement, use_spectral_attention, etc.)
- Compatible from_pretrained() accepting same parameters as sophisticated version
- save_pretrained() method for checkpointing

Training Script Integration
- Updated import to use benchmark model: AutoencoderMSBenchmark as AutoencoderKLMultispectralAdapter
- forward() method supports mask parameter and returns (reconstruction, losses) in training mode
- Supports spectral guidance for scientific realism
- Includes SAM loss for spectral fidelity
"""

from typing import Dict, Optional, Tuple, Union
import logging
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

# Set up logger for this module
logger = logging.getLogger(__name__)

# Numerically stable normalization for SAM loss (copied from sophisticated adapter)
def safe_normalize(tensor, dim=1, eps=1e-8):
    """Numerically stable normalization for SAM loss computation."""
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
        logger.debug("NaNs detected in SAM angle computation.")

    return angle.mean()

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
    Now includes SAM loss and spectral guidance for scientific comparison.
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
        use_sam_loss: bool = False,  # Now configurable
        adapter_in_channels: int = 5,
        adapter_out_channels: int = 5,
        backbone_in_channels: int = 3,
        backbone_out_channels: int = 3,
        use_saturation_penalty: bool = False,  # Disabled for benchmark
        spectral_guidance_weight: float = 0.05,  # NEW: weight for spectral guidance
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
        self.spectral_guidance_weight = spectral_guidance_weight
        
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
            spectral_guidance_weight=kwargs.get("spectral_guidance_weight", 0.05),
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
            spectral_guidance_weight=self.spectral_guidance_weight,
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

    def compute_losses(self, original: torch.Tensor, reconstructed: torch.Tensor, mask: torch.Tensor = None, 
                      reference_signature: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute loss terms for benchmark comparison with spectral guidance and SAM loss.
        
        Args:
            original: Original multispectral image, shape (B, 5, H, W)
            reconstructed: Reconstructed multispectral image, shape (B, 5, H, W)
            mask: Binary mask (1 for leaf/foreground, 0 for background), shape (B, 1, H, W)
            reference_signature: Reference spectral signature for guidance, shape (5,)
        
        Returns:
            Dictionary containing loss terms
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
            # No mask provided - use full image
            mask = torch.ones_like(original)
            losses['mask_stats'] = {
                'coverage': 1.0,
                'valid_pixels': mask.numel(),
                'total_pixels': mask.numel()
            }

        # Apply mask to both original and reconstructed images
        masked_original = original * mask
        masked_reconstructed = reconstructed * mask

        # Simple MSE loss with masking
        mse_per_pixel = F.mse_loss(masked_reconstructed, masked_original, reduction='none')
        
        # Average over masked regions only (excluding background)
        if mask is not None:
            # Sum over spatial dimensions, then divide by number of valid pixels per channel
            mse_per_channel = (mse_per_pixel * mask).sum(dim=(0, 2, 3)) / (mask.sum(dim=(0, 2, 3)) + 1e-8)
        else:
            mse_per_channel = mse_per_pixel.mean(dim=(0, 2, 3))
        
        losses['mse_per_channel'] = mse_per_channel
        losses['mse'] = mse_per_channel.mean()

        # SAM loss for spectral fidelity (if enabled)
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

        # Spectral guidance loss (copied from sophisticated adapter)
        if reference_signature is not None:
            # Compute mean spectrum over leaf pixels for guidance
            if mask is not None:
                # Use original mask shape (B, 1, H, W) for summing
                original_mask = mask[:, :1, :, :]  # Take first channel of expanded mask
                mask_sum = original_mask.sum(dim=(2, 3), keepdim=False)  # (B, 1)
                recon_sum = (reconstructed * mask).sum(dim=(2, 3))  # (B, 5)
                recon_mean_spectrum = recon_sum / (mask_sum + 1e-8)  # (B, 5)
            else:
                recon_mean_spectrum = reconstructed.mean(dim=(2, 3))  # (B, 5)
            
            # Ensure reference_signature is on the same device and has correct shape
            if reference_signature.device != reconstructed.device:
                reference_signature = reference_signature.to(reconstructed.device)
            
            # Expand reference to match batch size: (5,) -> (B, 5)
            if reference_signature.dim() == 1:
                reference_signature = reference_signature.unsqueeze(0).expand(reconstructed.shape[0], -1)
            
            # Compute guidance loss (MSE between mean reconstructed spectrum and reference)
            spectral_guidance_loss = F.mse_loss(recon_mean_spectrum, reference_signature)
            losses['spectral_guidance'] = spectral_guidance_loss
            
            # Log spectral guidance info
            logger.debug(f"[Spectral Guidance] Recon mean: {recon_mean_spectrum.mean(0).cpu().numpy()}")
            logger.debug(f"[Spectral Guidance] Reference: {reference_signature.mean(0).cpu().numpy()}")
            logger.debug(f"[Spectral Guidance] Loss: {spectral_guidance_loss.item():.6f}")

        # Total loss computation
        total_loss = losses['mse']
        
        # Add SAM loss if enabled
        if self.use_sam_loss and 'sam' in losses:
            total_loss = total_loss + 0.1 * losses['sam']  # Weight SAM loss
        
        # Add spectral guidance loss if available
        if reference_signature is not None and 'spectral_guidance' in losses:
            total_loss = total_loss + self.spectral_guidance_weight * losses['spectral_guidance']
        
        losses['total_loss'] = total_loss
        
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
        reference_signature: Optional[torch.Tensor] = None,
    ) -> Union[DecoderOutput, torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass with minimal 5-channel processing, SAM loss, and spectral guidance.
        
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

        # Training mode with loss computation
        if self.training and mask is not None:
            losses = self.compute_losses(x, dec, mask, reference_signature)
            return dec, losses
        
        # Inference mode
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec) 