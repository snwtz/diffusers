"""
Simple CLI-based inference for DreamBooth Multispectral Model

This script provides a simple command-line interface for generating images
from a trained DreamBooth multispectral model, similar to the original DreamBooth approach.

Usage:
    python inference_dreambooth_multispectral.py \
        --model_path "path/to/trained/model" \
        --prompt "sks leaf" \
        --output_dir "output_images"


--guidance_scale: Controls how strongly the prompt influences the image. Default is 7.5.
Higher values = more prompt adherence, lower = more creative.
--num_inference_steps: Number of denoising steps. Default is 50.
Higher = better quality, slower inference.
--seed: For reproducibility. Set a value to get the same images each run.
--prompt: You can change the prompt to test different concepts.
--use_best_model: If you want to use the best model checkpoint (if available) instead of the final model.
"""

import argparse
import os
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.models import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModel, T5EncoderModel
# Add import for custom VAE
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from diffusers.models.autoencoders.autoencoder_kl_multispectral_adapter import AutoencoderKLMultispectralAdapter
import numpy as np
import rasterio
from rasterio.transform import Affine
import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

def create_pseudo_rgb(bands):
    """
    Create pseudo-RGB visualization of multispectral images.
    
    Maps the 5 spectral bands to RGB channels for human visualization:
    - Band 1 (474.73nm - Blue) -> Blue channel
    - Band 2 (538.71nm - Green) -> Green channel  
    - Band 3 (650.665nm - Red) -> Red channel
    - Band 4 (730.635nm - Red-edge) -> Additional red contribution
    - Band 5 (850.59nm - NIR) -> Additional green contribution
    
    This mapping follows the biological relevance of each band for plant health analysis.
    """
    # Normalize from [-1, 1] to [0, 1] for RGB visualization
    bands_norm = (bands + 1) / 2
    
    # Create RGB channels with spectral mapping
    r = bands_norm[2] * 0.7 + bands_norm[3] * 0.3  # Red + Red-edge
    g = bands_norm[1] * 0.6 + bands_norm[4] * 0.4  # Green + NIR
    b = bands_norm[0]  # Blue
    
    # Stack channels and ensure proper range
    rgb = np.stack([r, g, b], axis=0)
    rgb = np.clip(rgb, 0, 1)
    return rgb

def create_pseudo_rgb_training_style(bands):
    """
    Create pseudo-RGB visualization from 5-channel multispectral data (training validation style).
    
    Maps specific spectral bands to RGB channels for human visualization:
    - Red channel: Band 2 (650.665nm - Red) 
    - Green channel: Band 1 (538.71nm - Green)
    - Blue channel: Band 0 (474.73nm - Blue)
    
    Args:
        bands: numpy array of shape (5, H, W) in [-1, 1] range
        
    Returns:
        numpy array of shape (3, H, W) in [0, 1] range
    """
    # Normalize from [-1, 1] to [0, 1] for RGB visualization
    bands_norm = (bands + 1) / 2
    
    # Create RGB channels using specific band mappings (same as training validation)
    r = bands_norm[2]  # Red channel (650.665nm)
    g = bands_norm[1]  # Green channel (538.71nm) 
    b = bands_norm[0]  # Blue channel (474.73nm)
    
    # Stack channels and ensure proper range
    rgb = np.stack([r, g, b], axis=0)
    rgb = np.clip(rgb, 0, 1)
    return rgb

def plot_individual_bands(bands, output_dir, image_idx):
    """
    Create individual plots for each of the 5 spectral bands.
    
    Args:
        bands: numpy array of shape (5, H, W) containing the 5 spectral bands
        output_dir: directory to save the plots
        image_idx: index of the generated image for naming
    """
    import matplotlib.pyplot as plt
    
    # Band information for descriptive titles
    band_info = [
        "Band 1 (474.73nm - Blue)",
        "Band 2 (538.71nm - Green)",
        "Band 3 (650.665nm - Red)", 
        "Band 4 (730.635nm - Red-edge)",
        "Band 5 (850.59nm - NIR)"
    ]
    
    # Create subplot for all bands
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for i in range(5):
        # Normalize from [-1, 1] to [0, 1] for visualization
        band_norm = (bands[i] + 1) / 2
        
        # Display with greyscale colormap for all bands
        axes[i].imshow(band_norm, cmap='gray', vmin=0, vmax=1)
        
        axes[i].set_title(band_info[i], fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'individual_bands_{image_idx+1}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved individual band plots: individual_bands_{image_idx+1}.png")

def load_model_components(model_path, vae_path=None, device="cuda", dtype=torch.float16, base_model_path="stabilityai/stable-diffusion-3-medium-diffusers"):
    """
    Load individual model components from the trained model directory.
    
    Args:
        model_path: Path to the trained model directory
        vae_path: Path to the multispectral VAE (optional)
        device: Device to load models on
        dtype: Data type for models
        base_model_path: Path to base model for text encoders and tokenizers
    
    Returns:
        tuple: (transformer, vae, text_encoder, text_encoder_2, text_encoder_3, tokenizer, tokenizer_2, tokenizer_3, scheduler)
    """
    print(f"Loading model components from: {model_path}")
    
    # Load transformer
    transformer_path = os.path.join(model_path, "transformer")
    if os.path.exists(transformer_path):
        print(f"Loading transformer from: {transformer_path}")
        transformer = SD3Transformer2DModel.from_pretrained(
            transformer_path,
            torch_dtype=dtype,
            use_safetensors=True
        ).to(device)
    else:
        raise ValueError(f"Transformer not found at {transformer_path}")
    
    # Load VAE
    if vae_path is not None:
        print(f"Loading custom multispectral VAE from: {vae_path}")
        vae = AutoencoderKLMultispectralAdapter.from_pretrained(vae_path).float()
        vae = vae.to(device, dtype=torch.float32)
        print("MS adapter loaded")
        if hasattr(vae, 'input_adapter'):
            print(f"   - Input adapter: {vae.input_adapter.in_channels} → {vae.input_adapter.out_channels} channels")
        if hasattr(vae, 'output_adapter'):
            print(f"   - Output adapter: {vae.output_adapter.in_channels} → {vae.output_adapter.out_channels} channels")
    else:
        # Load default VAE from model path
        vae_path = os.path.join(model_path, "vae")
        if os.path.exists(vae_path):
            print(f"Loading VAE from: {vae_path}")
            vae = AutoencoderKLMultispectralAdapter.from_pretrained(vae_path).to(device, dtype=torch.float32)
        else:
            raise ValueError(f"VAE not found at {vae_path}")
    
    # Load text encoders and tokenizers from base model (they're typically not saved in DreamBooth)
    print(f"Loading text encoders and tokenizers from base model: {base_model_path}")
    
    # Load text encoders
    from transformers import CLIPTextModelWithProjection, T5EncoderModel
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        base_model_path, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        base_model_path, subfolder="text_encoder_2", torch_dtype=dtype
    ).to(device)
    text_encoder_3 = T5EncoderModel.from_pretrained(
        base_model_path, subfolder="text_encoder_3", torch_dtype=dtype
    ).to(device)
    
    # Load tokenizers
    from transformers import CLIPTokenizer, T5TokenizerFast
    tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer_2")
    tokenizer_3 = T5TokenizerFast.from_pretrained(base_model_path, subfolder="tokenizer_3")
    
    # Load scheduler
    scheduler_path = os.path.join(model_path, "scheduler")
    if os.path.exists(scheduler_path):
        print(f"Loading scheduler from: {scheduler_path}")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_path)
    else:
        print("Scheduler not found, using default FlowMatchEulerDiscreteScheduler")
        scheduler = FlowMatchEulerDiscreteScheduler()
    
    return transformer, vae, text_encoder, text_encoder_2, text_encoder_3, tokenizer, tokenizer_2, tokenizer_3, scheduler

def main():
    parser = argparse.ArgumentParser(description="Inference for DreamBooth Multispectral Model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained DreamBooth model directory"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="sks leaf",
        help="Text prompt for image generation"
    )
    # Output directory will always be set based on model_path
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for classifier-free guidance"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--use_best_model",
        action="store_true",
        help="Use best_model checkpoint instead of final_model"
    )
    # Add argument for MS VAE
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help="Path to custom multispectral VAE directory (optional)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for generated images (default: model_path/inference_test)"
    )
    
    args = parser.parse_args()

    # Debug: Check CUDA availability and device info
    print("[DEBUG] torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[DEBUG] CUDA device count:", torch.cuda.device_count())
        print("[DEBUG] CUDA device name:", torch.cuda.get_device_name(0))
    else:
        print("[DEBUG] CUDA not available! Inference will be very slow.")

    # Determine model path
    if args.use_best_model:
        model_path = os.path.join(args.model_path, "best_model")
        if not os.path.exists(model_path):
            print(f"Best model not found at {model_path}, using final model")
            model_path = os.path.join(args.model_path, "final_model")
    else:
        model_path = os.path.join(args.model_path, "final_model")
    
    if not os.path.exists(model_path):
        print(f"Final model not found at {model_path}, using main model directory")
        model_path = args.model_path

    # Define output directory
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(model_path, "inference_test")
    
    print(f"Loading model from: {model_path}")
    print(f"Output directory: {output_dir}")
    
    # Set device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Load model components
    try:
        transformer, vae, text_encoder, text_encoder_2, text_encoder_3, tokenizer, tokenizer_2, tokenizer_3, scheduler = load_model_components(
            model_path, args.vae, device, pipeline_dtype
        )
    except Exception as e:
        print(f"Error loading model components: {e}")
        return

    # Create pipeline manually
    pipeline = StableDiffusion3Pipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        text_encoder_3=text_encoder_3,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        tokenizer_3=tokenizer_3,
        scheduler=scheduler,
    )

    # Patch the pipeline's image processor to handle multispectral output
    if hasattr(vae, 'input_adapter'):  # Check if it's multispectral VAE
        original_postprocess = pipeline.image_processor.postprocess
        def patched_postprocess(image, output_type="pil", **kwargs):
            # Convert 5-channel multispectral to RGB for display
            if image.shape[1] == 5:  # 5-channel multispectral
                print(f"[DEBUG] Converting 5-channel multispectral image to pseudo-RGB for display")
                # Store the multispectral image for later saving
                if not hasattr(pipeline, '_multispectral_outputs'):
                    pipeline._multispectral_outputs = []
                pipeline._multispectral_outputs.append(image.clone())
                
                # Convert to pseudo-RGB using training validation style mapping
                batch_size = image.shape[0]
                rgb_images = []
                for i in range(batch_size):
                    # Extract 5 bands for this sample: (5, H, W)
                    bands = image[i]  # Shape: (5, H, W)
                    # Move to CPU and convert to numpy for pseudo-RGB creation
                    bands_cpu = bands.cpu().numpy()
                    # Create pseudo-RGB: (3, H, W) - using training validation style
                    rgb = create_pseudo_rgb_training_style(bands_cpu)
                    rgb_images.append(torch.from_numpy(rgb).to(image.device))
                
                # Stack back to batch: (B, 3, H, W)
                image_rgb = torch.stack(rgb_images, dim=0)
                return original_postprocess(image_rgb, output_type=output_type, **kwargs)
            else:
                return original_postprocess(image, output_type=output_type, **kwargs)
        pipeline.image_processor.postprocess = patched_postprocess

    # Move pipeline to device
    pipeline = pipeline.to(device)

    # Set pipeline components to eval mode for inference
    pipeline.transformer.eval()
    if hasattr(pipeline, 'vae') and pipeline.vae is not None:
        pipeline.vae.eval()

    # Debug: Print pipeline device and dtype
    try:
        print("[DEBUG] Pipeline transformer device:", next(pipeline.transformer.parameters()).device)
        print("[DEBUG] Pipeline transformer dtype:", next(pipeline.transformer.parameters()).dtype)
    except Exception as e:
        print("[DEBUG] Could not get pipeline transformer device/dtype:", e)
    if hasattr(pipeline, 'vae') and pipeline.vae is not None:
        try:
            print("[DEBUG] Pipeline VAE device:", next(pipeline.vae.parameters()).device)
            print("[DEBUG] Pipeline VAE dtype:", next(pipeline.vae.parameters()).dtype)
        except Exception as e:
            print("[DEBUG] Could not get pipeline VAE device/dtype:", e)

    # Warn if running on CPU or float32
    if device == "cpu":
        print("[WARNING] Running on CPU! This will be extremely slow. Check your CUDA setup.")
    if next(pipeline.transformer.parameters()).dtype == torch.float32 and device == "cuda":
        print("[WARNING] Running on CUDA but using float32. Consider using float16 for better performance.")

    # Set generator for reproducibility
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        print(f"Using seed: {args.seed}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {args.num_images} images with prompt: '{args.prompt}'")

    images = []
    if args.num_images == 1:
        # Single image, use batch pipeline call
        result = pipeline(
            prompt=args.prompt,
            num_images_per_prompt=1,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
        images = result.images
    else:
        # Multiple images, generate one by one for memory efficiency
        for i in range(args.num_images):
            print(f"[DEBUG] Generating image {i+1}/{args.num_images}")
            # For reproducibility, advance the seed if set
            gen = None
            if generator is not None:
                gen = torch.Generator(device=device).manual_seed(args.seed + i)
            result = pipeline(
                prompt=args.prompt,
                num_images_per_prompt=1,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=gen,
            )
            images.extend(result.images)

    # Save images
    for i, image in enumerate(images):
        # This is the pipeline's default RGB output (pseudo-RGB conversion)
        filename = f"pipeline_rgb_output_{i+1}.png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
    
    # Save multispectral images if available
    if hasattr(pipeline, '_multispectral_outputs') and pipeline._multispectral_outputs:
        print(f"[DEBUG] Saving {len(pipeline._multispectral_outputs)} multispectral images")
        for i, ms_image in enumerate(pipeline._multispectral_outputs):
            # Convert to numpy: Shape: (1, 5, H, W)
            ms_numpy = ms_image.cpu().numpy()[0]  # Remove batch dimension: (5, H, W)
            
            # Create individual band plots
            plot_individual_bands(ms_numpy, output_dir, i)
            
            # Also save pseudo-RGB visualization (training validation style)
            rgb = create_pseudo_rgb_training_style(ms_numpy)  # Shape: (3, H, W)
            rgb_filename = f"training_style_pseudo_rgb_{i+1}.png"
            rgb_filepath = os.path.join(output_dir, rgb_filename)
            
            # Save pseudo-RGB as PNG
            from PIL import Image
            # Convert from (3, H, W) to (H, W, 3) and scale to 0-255
            rgb_display = (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)
            rgb_image = Image.fromarray(rgb_display)
            rgb_image.save(rgb_filepath)
            
            print(f"Saved training-style pseudo-RGB: {rgb_filepath} (shape: {rgb.shape})")
    
    print(f"Generated {len(images)} images in {output_dir}")
    print("\nFile descriptions:")
    print("- pipeline_rgb_output_*.png: Pipeline's default RGB output (training-style pseudo-RGB conversion)")
    print("- training_style_pseudo_rgb_*.png: Training validation style mapping (B0→B, B1→G, B2→R)")
    print("- individual_bands_*.png: Individual plots of each of the 5 spectral bands")

if __name__ == "__main__":
    main()