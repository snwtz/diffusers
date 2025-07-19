"""
Simple CLI-based inference for DreamBooth Multispectral Model

This script provides a simple command-line interface for generating images
from a trained DreamBooth multispectral model, similar to the original DreamBooth approach.

Usage:
    python inference_dreambooth_multispectral.py \
        --model_path "path/to/trained/model" \
        --prompt "sks leaf" \
        --output_dir "output_images"
"""

import argparse
import os
import torch
from diffusers import StableDiffusion3Pipeline

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
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_output",
        help="Directory to save generated images"
    )
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
    
    args = parser.parse_args()
    
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
    
    print(f"Loading model from: {model_path}")
    
    # Load pipeline
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True
    )
    
    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    
    # Set generator for reproducibility
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        print(f"Using seed: {args.seed}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating {args.num_images} images with prompt: '{args.prompt}'")
    
    # Generate images
    images = pipeline(
        prompt=args.prompt,
        num_images_per_prompt=args.num_images,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images
    
    # Save images
    for i, image in enumerate(images):
        filename = f"generated_{i+1}.png"
        filepath = os.path.join(args.output_dir, filename)
        image.save(filepath)
        print(f"Saved: {filepath}")
    
    print(f"Generated {len(images)} images in {args.output_dir}")

if __name__ == "__main__":
    main() 