# Masked Loss Implementation for Pure Leaf-Focused Training

## Overview

This document describes the implementation of masked loss computation in the multispectral VAE adapter, enabling pure leaf-focused training by excluding background regions from loss calculations.

## Key Features

### 1. Pure Leaf-Focused Loss Computation

The `compute_losses()` method now implements comprehensive masking:

- **Binary Masking**: Uses binary masks (1 for leaf, 0 for background) to focus training
- **Per-Channel MSE**: Computes MSE loss only on masked leaf regions
- **Masked SAM Loss**: Applies Spectral Angle Mapper loss only on valid spectral pixels
- **Coverage Monitoring**: Tracks mask coverage statistics for training stability

### 2. Enhanced Forward Method

The `forward()` method now supports mask parameter:

```python
# Training with mask
reconstruction, losses = model.forward(sample=batch, mask=mask)

# Evaluation without mask (optional)
reconstruction, losses = model.forward(sample=batch, mask=None)
```

### 3. Comprehensive Mask Validation

- **Shape Validation**: Ensures mask compatibility with input dimensions
- **Binary Enforcement**: Converts masks to binary (0 or 1) values
- **Coverage Warnings**: Alerts for very low or high mask coverage
- **Fallback Handling**: Graceful handling when no mask is provided

## Implementation Details

### Mask Processing

```python
# Validate and prepare mask for broadcasting
if mask.shape[1] == 1:
    mask = mask.expand(-1, original.shape[1], -1, -1)  # (B, 1, H, W) -> (B, 5, H, W)
mask = (mask > 0.5).float()  # Ensure binary values
```

### Masked MSE Loss

```python
# Apply mask to both original and reconstructed images
masked_original = original * mask
masked_reconstructed = reconstructed * mask

# Compute MSE only on masked regions
mse_per_pixel = F.mse_loss(masked_reconstructed, masked_original, reduction='none')
mse_per_channel = (mse_per_pixel * mask).sum(dim=(0, 2, 3)) / (mask.sum(dim=(0, 2, 3)) + 1e-8)
```

### Masked SAM Loss

```python
# Compute spectral magnitude for valid pixel detection
spectral_magnitude = torch.norm(masked_original, dim=1, keepdim=True)
valid_spectral_mask = (spectral_magnitude > 1e-6) & (mask[:, :1, :, :] > 0.5)

# Extract valid pixels for SAM computation
valid_original = masked_original[valid_spectral_mask.expand_as(masked_original)]
valid_reconstructed = masked_reconstructed[valid_spectral_mask.expand_as(masked_reconstructed)]
```

## Training Integration

### Updated Training Script

The training script now uses the simplified forward method:

```python
# Before (manual encoding/decoding)
posterior = model.encode(batch).latent_dist
z = posterior.sample()
reconstruction = model.decode(z, return_dict=False)
losses = model.compute_losses(batch, reconstruction, mask=mask)

# After (simplified forward)
reconstruction, losses = model.forward(sample=batch, mask=mask)
```

### Enhanced Logging

Mask statistics are now logged during training:

```python
# Mask coverage monitoring
if 'mask_stats' in losses:
    mask_coverage = losses['mask_stats']['coverage']
    if mask_coverage < 0.1:
        logger.warning(f"Low mask coverage: {mask_coverage:.4f}")
```

## Benefits

### 1. Training Efficiency

- **Focused Learning**: Model learns only from biologically relevant regions
- **Reduced Noise**: Background artifacts don't interfere with training
- **Faster Convergence**: Cleaner gradients from leaf-only loss computation

### 2. Scientific Validity

- **Plant-Specific Training**: Ensures model focuses on leaf spectral signatures
- **Background Independence**: Eliminates background bias in spectral learning
- **Reproducible Results**: Consistent training regardless of background variations

### 3. Monitoring and Debugging

- **Coverage Tracking**: Monitor mask coverage throughout training
- **Loss Transparency**: Clear separation of leaf vs. background contributions
- **Early Warning**: Alerts for problematic mask coverage

## Usage Examples

### Basic Training

```python
# Initialize model
model = AutoencoderKLMultispectralAdapter.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    adapter_placement="both",
    use_spectral_attention=True,
    use_sam_loss=True
)

# Training loop
for batch, mask in dataloader:
    reconstruction, losses = model.forward(sample=batch, mask=mask)
    total_loss = losses['mse']  # Already masked
    total_loss.backward()
```

### Monitoring Mask Statistics

```python
# Check mask coverage
if 'mask_stats' in losses:
    stats = losses['mask_stats']
    print(f"Coverage: {stats['coverage']:.4f}")
    print(f"Valid pixels: {stats['valid_pixels']}/{stats['total_pixels']}")
```

### Testing

Run the test script to verify implementation:

```bash
python examples/multispectral/test_masked_loss.py
```

## Compatibility

### Backward Compatibility

- **Optional Masking**: Training works with or without masks
- **Fallback Behavior**: Graceful handling when no mask is provided
- **Existing Code**: No breaking changes to existing training pipelines

### Forward Compatibility

- **Enhanced Monitoring**: New mask statistics for better training insight
- **Simplified API**: Cleaner forward method interface
- **Extensible Design**: Easy to add new mask-based features

## Scientific Impact

This implementation directly supports the thesis goals by:

1. **Ensuring Spectral Fidelity**: Focused training on leaf regions preserves spectral signatures
2. **Improving Model Performance**: Cleaner training data leads to better spectral reconstruction
3. **Supporting Plant Science**: Model learns plant-specific spectral patterns
4. **Enabling Reproducible Research**: Consistent training regardless of background variations

The masked loss computation represents a significant methodological contribution to multispectral plant image generation, ensuring that the model learns the most relevant spectral information for plant health analysis. 