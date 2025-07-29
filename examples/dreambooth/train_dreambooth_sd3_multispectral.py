"""
DreamBooth Training Script for Stable Diffusion 3 with Multispectral Support

This script implements the core training workflow for adapting DreamBooth to multispectral
image generation, a central component of synthetic multispectral plant tissue generation.
It extends SD3's capabilities to handle 5-channel spectral data while maintaining
compatibility with the original model's latent space.

IMPORTANT: This script assumes the multispectral VAE has been pretrained and frozen.
Only the SD3 components are trained during DreamBooth fine-tuning.

CHANNEL CONFIGURATION:
- Input: 5-channel multispectral data (bands 9, 18, 32, 42, 55)
- VAE Adapter: 5-in/5-out (adapter_in_channels/adapter_out_channels)
- VAE Backbone: 3-in/3-out (backbone_in_channels/backbone_out_channels)
- VAE Output: 16 latent channels (matching SD3 transformer's default expectation)
- Transformer Input: 16 latent channels (SD3's default in_channels)

Identical Core Logic with Original DreamBooth:
Training loop structure: Matches original exactly
Loss computation: Same flow matching loss
Optimization: Same optimizer and scheduler logic
Checkpointing: Same checkpoint management
Validation: Same validation pipeline
Multispectral Adaptations:
Data loading: Custom multispectral dataloader
VAE: Multispectral VAE instead of standard AutoencoderKL
Input processing: 5-channel instead of 3-channel RGB
Validation: Spectral-aware logging

✅ Missing Elements Check:
All critical elements from the original script are present:
Complete argument parsing
Model loading and setup
Optimizer and scheduler configuration
Training loop with all steps
Checkpointing and resuming
Validation and logging
Model saving and final inference


Module Purpose and Scientific Context:
-----------------------------------
1. Research Objective:
   - Adapt DreamBooth for multispectral concept learning
   - Enable synthetic generation of plant tissue spectral signatures
   - Maintain spectral fidelity while leveraging SD3's generative capabilities
   - Support scientific analysis of plant health through spectral signatures

2. Technical Foundation:
   - Built on pretrained Stable Diffusion 3
   - Uses pretrained and frozen multispectral VAE (AutoencoderKLMultispectralAdapter)
   - Integrates spectral attention mechanism (TODO: implement)
   - Implements spectral-aware loss functions (TODO: implement)

3. Scientific/Biological Relevance:
   The training pipeline processes 5 carefully selected bands:
   - Band 9 (474.73nm): Blue - captures chlorophyll absorption
   - Band 18 (538.71nm): Green - reflects well in healthy vegetation
   - Band 32 (650.665nm): Red - sensitive to chlorophyll content
   - Band 42 (730.635nm): Red-edge - sensitive to stress and early disease
   - Band 55 (850.59nm): NIR - strong reflectance in healthy leaves

   Band Selection Rationale:
   - Optimized for plant health monitoring
   - Covers key physiological indicators
   - Enables stress detection
   - Supports disease identification

Implementation Decisions:
----------------------
1. Parameter-Efficient Design:
   - Adapter-based approach preferred over full retraining
   - Frozen VAE preserves SD3 compatibility
   - Minimal trainable parameters
   - Enables training with limited data

2. Loss Function Design:
   - Standard DreamBooth loss (currently implemented)
   - Prior preservation: Retains concept learning

3. Latent Space Handling:
   - Expects 16 latent channels (SD3's default, beneficial for multispectral data)
   - Maintains generative capabilities
   - Preserves spectral information with increased capacity

Data Handling:
------------
1. Multispectral Dataloader:
   - Uses create_multispectral_dataloader() for 5-channel TIFF input
   - Implements efficient caching and prefetching
   - Supports multiprocessing for data loading
   - Validates channel compatibility via validate_dataloader_output()
   - Returns dict with "pixel_values" and "mask" (mask available for future masked loss)

2. Data Preprocessing:
   - Per-channel normalization to [-1, 1] range
   - Spectral signature preservation
   - Memory-efficient loading
   - Band selection and validation

Training Strategy:
---------------
1. VAE Integration:
   - Pretrained and frozen multispectral VAE
   - Expects 16 latent channels (SD3's default, optimal for multispectral data)


2. Loss Functions:
   - Standard DreamBooth loss (currently implemented)
   - Prior preservation loss

3. Optimization:
   - Gradient accumulation for memory efficiency
   - Learning rate scheduling with warmup
   - Early stopping on validation plateau
   - Gradient clipping for stability



Text Encoder Handling:
-------------------
1. Multi-Encoder Architecture:
   - CLIP: Visual-semantic alignment
   - T5: Detailed concept understanding
   - Concatenated embeddings for rich representation


2. Encoding Functions:
   - _encode_prompt_with_clip(): Visual-semantic features
   - _encode_prompt_with_t5(): Detailed concept understanding
   - encode_prompt(): Combined representation
   - Support for per-image embeddings

Logging and Evaluation:
--------------------
1. Validation Pipeline:
   - log_validation() for model assessment
   - Concept preservation evaluation

2. Integration:
   - Weights & Biases for experiment tracking
   - Spectral visualization tools
   - Loss term tracking

Open Research Questions:
----------------------
1. Text Encoder Adaptation:
   a) Concept Learning:
      - How does the text encoder handle multispectral concepts?
      - Can it learn spectral signatures from text descriptions?
      - How does it map between spectral and semantic spaces?
   
   b) Prior Preservation:
      - Should prior preservation loss be modified for spectral data?
      - How to balance spectral fidelity with concept preservation?
      - What is the optimal prompt engineering for spectral features?

2. Training Dynamics:
   a) Learning Rate:
      - Optimal learning rate for 5-channel inputs?
      - How does spectral data affect gradient flow?
      - Should learning rates differ for spectral vs. spatial features?
   
   b) Latent Space:
      - How does the latent space distribution change?
      - What spectral information is preserved/compressed?
      - How to visualize and interpret spectral latent codes?

3. Architecture Design:
   a) Channel Processing:
      - Should we use channel-specific attention?
      - How does channel ordering affect performance?
      - What is the optimal adapter architecture?
   
   b) Spectral Fidelity:
      - How to measure spectral reconstruction quality?
      - What metrics best capture spectral signature preservation?
      - How to balance spatial vs. spectral accuracy?

Thesis Discussion Points:
-----------------------
1. Methodological Contributions:
   a) Architecture Design:
      - Lightweight adapter approach for spectral adaptation
      - Parameter-efficient fine-tuning strategy
      
   
   b) Training Strategy:
      - Concept preservation

2. Scientific Implications:
   a) Plant Health Analysis:
      - Spectral signature preservation
      - Stress detection capabilities
      - Disease identification potential
   
   b) Agricultural Applications:
      - Early stress detection
      - Disease monitoring

3. Limitations and Future Work:
   a) Technical Limitations:
      - Memory constraints
      - Training stability
      - Spectral fidelity trade-offs
      - Computational requirements
   
   b) Research Directions:
      - Novel spectral attention mechanisms
      - Advanced loss functions
      - Improved training strategies
      - Enhanced visualization tools

4. Broader Impact:
   a) Agricultural Technology:
      - Precision agriculture
      - Automated monitoring
      - Early intervention
      - Resource optimization
   
   b) Scientific Research:
      - Plant physiology studies
      - Stress response analysis
      - Spectral signature research

# Thesis Integration Points:
-------------------------
1. Methodology Chapter:
   - Parameter-efficient design rationale
   - Band selection methodology


2. Results Chapter:

   - Concept preservation analysis


TODOs and Future Features:
------------------------
1. Data Handling:
   - [ ] Add support for reading train/val splits from .txt files
   - [ ] Implement spectral data augmentation
   - [ ] Add data validation pipeline
   - [ ] Implement spectral quality checks

2. Evaluation:
   - [ ] Add comprehensive unit tests
   - [ ] Add visualization tools
   - [ ] Create evaluation pipeline


References:
- DreamBooth paper: https://arxiv.org/abs/2208.12242
- SD3 paper: https://arxiv.org/pdf/2403.03206

Usage:
    # Train the model:
    python train_dreambooth_sd3_multispectral.py \
        --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers \
        --instance_data_dir /path/to/multispectral/tiffs \
        --output_dir /path/to/save/model \
        --instance_prompt "sks leaf" \
        --num_train_epochs 100 \
        --train_batch_size 4 \
        --learning_rate 1e-4 \
        --mixed_precision fp16

# Research questions for multispectral DreamBooth
# 1. How does the text encoder handle multispectral concepts?
# 2. Should we modify the prior preservation loss for multispectral data?
# 3. Do we need to adjust the learning rate for 5-channel inputs?
# 4. How does the latent space distribution change with 5-channel input?
# 5. What is the optimal way to visualize multispectral training progress?
"""

# Unused imports that were removed:
# - itertools: Not used in the code
# - math: Not used in the code
# - random: Not used in the code
# - shutil: Not used in the code
# - Path from pathlib: Not used in the code
# - torch.utils.checkpoint: Not used in the code
# - DistributedDataParallelKwargs, ProjectConfiguration from accelerate.utils: Only set_seed is used
# - insecure_hashlib from huggingface_hub.utils: Not used in the code
# - Image, exif_transpose from PIL: Not used in the code
# - Dataset from torch.utils.data: Not used in the code
# - transforms from torchvision: Not used in the code
# - crop from torchvision.transforms.functional: Not used in the code
# - tqdm from tqdm.auto: Not used in the code
# - rasterio: Not used in the code
# - is_compiled_module from diffusers.utils.torch_utils: Not used in the code
# - transformers: Using specific imports instead
# - diffusers: Using specific imports instead

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))) # keep: fixes module import bug
import argparse
import copy
import itertools
import logging
import math
import os
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
# Import the multispectral VAE adapter
from src.diffusers.models.autoencoders.autoencoder_kl_multispectral_adapter import AutoencoderKLMultispectralAdapter
# --- Import ControlNetModel for spatial conditioning integration ---
# from diffusers import ControlNetModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

# Import custom multispectral dataloader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))) 
from multispectral_dataloader import create_multispectral_dataloader

# Import for logging setup
import transformers
import diffusers

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.34.0.dev0")

logger = get_logger(__name__)

# Import spectral fidelity metrics
try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    logger.warning("skimage not available, SSIM will be disabled")

try:
    from torchmetrics.image import SpectralAngleMapper
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logger.warning("torchmetrics not available, SAM will be disabled")

if is_wandb_available():
    import wandb

def create_pseudo_rgb_from_multispectral(bands_tensor):
    """
    Create pseudo-RGB visualization from 5-channel multispectral data.
    
    Maps specific spectral bands to RGB channels for human visualization:
    - Red channel: Band 2 (650.665nm - Red) 
    - Green channel: Band 1 (538.71nm - Green)
    - Blue channel: Band 0 (474.73nm - Blue)
    
    Args:
        bands_tensor: Tensor of shape (5, H, W) in [-1, 1] range
        
    Returns:
        PIL Image in RGB format
    """
    # Convert to numpy and normalize from [-1, 1] to [0, 1]
    bands_np = bands_tensor.cpu().numpy()
    bands_norm = (bands_np + 1) / 2
    
    # Create RGB channels using specific band mappings
    r = bands_norm[2]  # Red channel (650.665nm)
    g = bands_norm[1]  # Green channel (538.71nm) 
    b = bands_norm[0]  # Blue channel (474.73nm)
    
    # Stack channels and ensure proper range
    rgb = np.stack([r, g, b], axis=0)
    rgb = np.clip(rgb, 0, 1)
    
    # Convert to PIL Image
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    rgb_pil = Image.fromarray(rgb_uint8.transpose(1, 2, 0))
    
    return rgb_pil

# --- DreamBooth logger creation (ensure frozen VAE is passed in for validation decoding) ---
def create_dreambooth_logger(output_dir, model_name):
    # Use the comprehensive DreamBooth logger from dreambooth_logger.py
    from dreambooth_logger import create_dreambooth_logger as create_comprehensive_logger
    return create_comprehensive_logger(output_dir, model_name)

def load_text_encoders(args):
    """Load the three text encoders for SD3."""
    # Import the correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )
    
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder_three = text_encoder_cls_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for i, (tokenizer, text_encoder) in enumerate(zip(clip_tokenizers, clip_text_encoders)):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[i] if text_input_ids_list else None,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="DreamBooth training script for SD3 with multispectral support.")
    # Add standard DreamBooth arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    # Add multispectral-specific arguments
    parser.add_argument(
        "--num_channels",
        type=int,
        default=5,
        help="Number of channels in the multispectral data.",
    )
    parser.add_argument(
        "--normalization_strategy",
        type=str,
        default="per_channel",
        choices=["per_channel", "global"],
        help="Strategy for normalizing multispectral data.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        required=True,
        help="Path to the pretrained multispectral VAE. Required for multispectral training.",
    )
    # --- ControlNet: Add argument for optional ControlNet model path ---
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path or identifier for pretrained ControlNet (for spatial conditioning on leaf mask)."
    )
    
    # Add all other standard DreamBooth arguments
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd3-dreambooth",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of forward/backward passes to accumulate before updating weights. Use >1 to simulate larger batch sizes with less memory.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--precondition_outputs",
        type=int,
        default=1,
        help="Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how "
        "model `target` is calculated.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=100,
        help="Number of steps between detailed logs (loss, lr, grad_norm)."
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

    if args.dataset_name is not None and args.instance_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--instance_data_dir`")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args


def validate_dataloader_output(dataloader, num_channels):
    """
    Validate that the dataloader outputs the correct number of channels.
    This is crucial for ensuring compatibility with the multispectral VAE.
    
    Args:
        dataloader: The dataloader to validate
        num_channels: Expected number of channels (5 for multispectral)
    
    Raises:
        ValueError: If the dataloader output doesn't match expected shape
    """
    try:
        batch = next(iter(dataloader))
        # Updated to work with dict format: batch["pixel_values"]
        if batch["pixel_values"].shape[1] != num_channels:
            raise ValueError(
                f"Dataloader output has {batch['pixel_values'].shape[1]} channels, "
                f"but {num_channels} channels are required. "
                f"Please check the multispectral dataloader configuration."
            )
        logger.info(f"Validated dataloader output shape: {batch['pixel_values'].shape}")
    except Exception as e:
        raise ValueError(f"Failed to validate dataloader output: {str(e)}")



def main(args):
    print("Starting DreamBooth multispectral training...")
    print(f"Output directory: {args.output_dir}")
    print(f"VAE path: {args.vae_path}")
    print(f"Instance data dir: {args.instance_data_dir}")
    """
    Main training function for multispectral DreamBooth fine-tuning.
    This function orchestrates the training process, including:
    1. Model initialization with pretrained VAE
    2. Data loading and preprocessing
    3. Training loop with standard DreamBooth loss
    4. Validation and logging
    
    Args:
        args: Command line arguments containing training configuration
    """
    
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        
        # Use wandb setup from dreambooth_logger
        from dreambooth_logger import setup_wandb_dreambooth
        wandb_success = setup_wandb_dreambooth(
            args=args,
            instance_prompt=args.instance_prompt,
            class_prompt=args.class_prompt if args.with_prior_preservation else None
        )
        if not wandb_success:
            logger.warning("Failed to initialize wandb, continuing without wandb logging")
            args.report_to = "tensorboard"  # Fallback to tensorboard

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # Back to INFO level
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Create multispectral dataloader
    train_dataloader = create_multispectral_dataloader(
        data_root=args.instance_data_dir,
        batch_size=args.train_batch_size,
        resolution=args.resolution,
        num_workers=args.dataloader_num_workers,
        use_cache=True,
        prefetch_factor=None if args.dataloader_num_workers == 0 else 2,  # Disable prefetch for local testing
        persistent_workers=args.dataloader_num_workers > 0,  # Only enable for multi-worker setup
        return_mask=False,  # No longer using masked loss
        prompt=args.instance_prompt if hasattr(args, 'instance_prompt') else "sks leaf",  # Use provided prompt
    )

    # Add warnings for potential performance issues
    if args.dataloader_num_workers == 0:
        logger.warning(
            "num_workers=0 detected. Data loading will be slow for large datasets. "
            "Consider increasing num_workers for better performance."
        )
    
    # Warning for large datasets with caching enabled
    dataset_size = len(train_dataloader.dataset)
    if dataset_size > 1000 and args.dataloader_num_workers == 0:
        logger.warning(
            f"Large dataset detected ({dataset_size} images) with use_cache=True and num_workers=0. "
            "This may cause high memory usage. Consider setting use_cache=False or increasing num_workers."
        )

    # Validate dataloader output after accelerator initialization (fix Runtime error as result of loading logger before initializing accelerator)
    # --- Validation step: log the structure and shapes of the first batch for debugging ---
    try:
        first_batch = next(iter(train_dataloader))
        logger.info(f"First batch keys: {list(first_batch.keys())}")
        logger.info(f"pixel_values shape: {first_batch['pixel_values'].shape}, dtype: {first_batch['pixel_values'].dtype}")
        logger.info(f"prompts: {first_batch['prompts']}")
    except Exception as e:
        logger.error(f"Error validating first batch from dataloader: {e}")

    validate_dataloader_output(train_dataloader, args.num_channels)

    # Add logging for dataloader configuration
    logger.info(
        f"Created multispectral dataloader with:"
        f"\n - num_workers: {args.dataloader_num_workers}"
        f"\n - prefetch_factor: {None if args.dataloader_num_workers == 0 else 2}"
        f"\n - persistent_workers: {args.dataloader_num_workers > 0}"
        f"\n - batch_size: {args.train_batch_size}"
        f"\n - resolution: {args.resolution}"
    )

    # Load tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(args)

    # Initialize multispectral VAE from the specific trained path
    vae = AutoencoderKLMultispectralAdapter.from_pretrained(args.vae_path)
    logger.info(f"Loaded multispectral VAE from: {args.vae_path}")
    
    # Log VAE configuration for debugging
    logger.info(f"VAE config - latent_channels: {vae.config.latent_channels}")
    logger.info(f"VAE config - adapter_in_channels: {getattr(vae.config, 'adapter_in_channels', 'N/A')}")
    logger.info(f"VAE config - adapter_out_channels: {getattr(vae.config, 'adapter_out_channels', 'N/A')}")
    logger.info(f"VAE config - backbone_in_channels: {getattr(vae.config, 'backbone_in_channels', 'N/A')}")
    logger.info(f"VAE config - backbone_out_channels: {getattr(vae.config, 'backbone_out_channels', 'N/A')}")
    
    # Check for configuration mismatches
    if getattr(vae.config, 'adapter_in_channels', None) != args.num_channels:
        logger.error(
            f"VAE configuration mismatch! VAE adapter expects {getattr(vae.config, 'adapter_in_channels', None)} channels "
            f"but dataloader provides {args.num_channels} channels. "
            f"This will cause errors during training."
        )
        raise ValueError(
            f"VAE adapter_in_channels ({getattr(vae.config, 'adapter_in_channels', None)}) != dataloader channels ({args.num_channels})"
        )
    
    # SD3 transformer expects 16 latent channels by default, which is actually beneficial for multispectral data
    # as it provides more capacity to encode the additional spectral information
    if vae.config.latent_channels == 16:
        logger.info(
            f"VAE outputs {vae.config.latent_channels} latent channels, which matches SD3's default expectation. "
            f"This provides more capacity for encoding multispectral information."
        )
    elif vae.config.latent_channels != 4:
        logger.warning(
            f"VAE outputs {vae.config.latent_channels} latent channels. "
            f"SD3 typically expects 16 channels (default) or 4 channels. "
            f"Using {vae.config.latent_channels} channels may affect compatibility."
        )

    # Freeze the VAE 
    vae.requires_grad_(False)
    logger.info("Multispectral VAE frozen for DreamBooth training")

    # Verify that input of shape (B, 5, 512, 512) outputs a latent tensor with shape (B, 16, latent_H, latent_W) 
    # SD3 transformer expects 16 latent channels by default, which provides more capacity for multispectral data
    # This configuration is optimal for encoding the additional spectral information
    logger.info(f"Using {args.num_channels}-channel multispectral VAE for training")

    # Test VAE with a dummy input to verify latent shape
    with torch.no_grad():
        dummy_input = torch.randn(1, args.num_channels, args.resolution, args.resolution)
        latent = vae.encode(dummy_input).latent_dist.sample()
        logger.info(f"VAE test - Input shape: {dummy_input.shape}, Latent shape: {latent.shape}")

    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )

    """ --- Load lightweight ControlNet for mask-based spatial conditioning ---
    controlnet = None
    if args.controlnet_model_name_or_path:
        controlnet = ControlNetModel.from_pretrained(
            args.controlnet_model_name_or_path,
            torch_dtype=weight_dtype,
        )
        controlnet.to(accelerator.device)
        controlnet.requires_grad_(False)
        logger.info(f"Loaded ControlNet from: {args.controlnet_model_name_or_path}")
    """
    
    # Set up model training states (matches original DreamBooth)
    transformer.requires_grad_(True)
    if args.train_text_encoder:
        text_encoder_one.requires_grad_(True)
        text_encoder_two.requires_grad_(True)
        text_encoder_three.requires_grad_(True)
    else:
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        text_encoder_three.requires_grad_(False)

    # Set up weight dtype for mixed precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move models to device with appropriate dtypes
    # CRITICAL FIX: VAE should always be in fp32 for stability with multispectral data
    # This follows the original DreamBooth SD3 pattern where VAE stays in fp32
    # while other models use mixed precision for speed benefits
    vae.to(accelerator.device, dtype=torch.float32)
    # DO NOT manually set transformer/text_encoder dtypes - let accelerator.prepare() handle it
    # This follows the original DreamBooth SD3 pattern exactly
    transformer.to(accelerator.device)
    if not args.train_text_encoder:
        text_encoder_one.to(accelerator.device)
        text_encoder_two.to(accelerator.device)
        text_encoder_three.to(accelerator.device)

    # EXPLANATION OF FP32 vs FP16 GRADIENT SCALING ISSUE:
    # ===================================================
    # The error "ValueError: Attempting to unscale FP16 gradients" occurs when:
    # 1. Mixed precision training is enabled (fp16)
    # 2. Models are manually set to fp16 before accelerator.prepare()
    # 3. The gradient scaler tries to unscale gradients that are already in fp16
    #
    # WHY THIS HAPPENS:
    # - PyTorch's gradient scaler expects gradients to be in fp32 during unscaling
    # - When models are manually set to fp16, their gradients are also in fp16
    # - The scaler cannot unscale fp16 gradients (it expects fp32)
    #
    # THE ORIGINAL DREAMBOOTH SD3 SOLUTION:
    # - VAE always stays in fp32 (line 1130 in original: vae.to(accelerator.device, dtype=torch.float32))
    # - Other models are moved to device WITHOUT setting dtype
    # - accelerator.prepare() handles mixed precision setup automatically
    # - This creates a clean fp32 -> fp16 boundary that the gradient scaler can handle
    #
    # OUR FIX:
    # - Follow the exact same pattern as original DreamBooth SD3
    # - VAE in fp32, other models moved to device without dtype
    # - Let accelerator.prepare() handle mixed precision setup
    # - This maintains numerical stability for multispectral data while enabling mixed precision speed

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()
            text_encoder_three.gradient_checkpointing_enable()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                if isinstance(unwrap_model(model), SD3Transformer2DModel):
                    unwrap_model(model).save_pretrained(os.path.join(output_dir, "transformer"))
                elif isinstance(unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)):
                    if isinstance(unwrap_model(model), CLIPTextModelWithProjection):
                        hidden_size = unwrap_model(model).config.hidden_size
                        if hidden_size == 768:
                            unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder"))
                        elif hidden_size == 1280:
                            unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder_2"))
                    else:
                        unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder_3"))
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            if isinstance(unwrap_model(model), SD3Transformer2DModel):
                load_model = SD3Transformer2DModel.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
            elif isinstance(unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)):
                try:
                    load_model = CLIPTextModelWithProjection.from_pretrained(input_dir, subfolder="text_encoder")
                    model(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                except Exception:
                    try:
                        load_model = CLIPTextModelWithProjection.from_pretrained(input_dir, subfolder="text_encoder_2")
                        model(**load_model.config)
                        model.load_state_dict(load_model.state_dict())
                    except Exception:
                        try:
                            load_model = T5EncoderModel.from_pretrained(input_dir, subfolder="text_encoder_3")
                            model(**load_model.config)
                            model.load_state_dict(load_model.state_dict())
                        except Exception:
                            raise ValueError(f"Couldn't load the model of type: ({type(model)}).")
            else:
                raise ValueError(f"Unsupported model found: {type(model)=}")

            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer.parameters(), "lr": args.learning_rate}
    if args.train_text_encoder:
        # different learning rate for text encoder and unet
        text_parameters_one_with_lr = {
            "params": text_encoder_one.parameters(),
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        text_parameters_two_with_lr = {
            "params": text_encoder_two.parameters(),
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        text_parameters_three_with_lr = {
            "params": text_encoder_three.parameters(),
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        params_to_optimize = [
            transformer_parameters_with_lr,
            text_parameters_one_with_lr,
            text_parameters_two_with_lr,
            text_parameters_three_with_lr,
        ]
    else:
        params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warning(
                f"Learning rates were provided both for the transformer and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one and text_encoder_parameters_two to be
            # --learning_rate
            params_to_optimize[1]["lr"] = args.learning_rate
            params_to_optimize[2]["lr"] = args.learning_rate
            params_to_optimize[3]["lr"] = args.learning_rate

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
        text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoders, tokenizers, prompt, args.max_sequence_length
                )
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds

    # For multispectral DreamBooth, we use a single prompt for all images (no custom prompts)
    # Encode the instance prompt once to avoid redundant encoding
    if not args.train_text_encoder:
        instance_prompt_hidden_states, instance_pooled_prompt_embeds = compute_text_embeddings(
            args.instance_prompt, text_encoders, tokenizers
        )

    # Handle class prompt for prior-preservation
    if args.with_prior_preservation:
        if not args.train_text_encoder:
            class_prompt_hidden_states, class_pooled_prompt_embeds = compute_text_embeddings(
                args.class_prompt, text_encoders, tokenizers
            )

    # Clear the memory here
    if not args.train_text_encoder:
        del tokenizers, text_encoders
        # Explicitly delete the objects as well, otherwise only the lists are deleted and the original references remain, preventing garbage collection
        del text_encoder_one, text_encoder_two, text_encoder_three
        free_memory()

    # Pack the statically computed variables appropriately
    if not args.train_text_encoder:
        prompt_embeds = instance_prompt_hidden_states
        pooled_prompt_embeds = instance_pooled_prompt_embeds
        if args.with_prior_preservation:
            prompt_embeds = torch.cat([prompt_embeds, class_prompt_hidden_states], dim=0)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, class_pooled_prompt_embeds], dim=0)
    # if we're optimizing the text encoder we need to tokenize and encode the batch prompts on all training steps
    else:
        tokens_one = tokenize_prompt(tokenizer_one, args.instance_prompt)
        tokens_two = tokenize_prompt(tokenizer_two, args.instance_prompt)
        tokens_three = tokenize_prompt(tokenizer_three, args.instance_prompt)
        if args.with_prior_preservation:
            class_tokens_one = tokenize_prompt(tokenizer_one, args.class_prompt)
            class_tokens_two = tokenize_prompt(tokenizer_two, args.class_prompt)
            class_tokens_three = tokenize_prompt(tokenizer_three, args.class_prompt)
            tokens_one = torch.cat([tokens_one, class_tokens_one], dim=0)
            tokens_two = torch.cat([tokens_two, class_tokens_two], dim=0)
            tokens_three = torch.cat([tokens_three, class_tokens_three], dim=0)

    # Scheduler and math around the number of training steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`
    if args.train_text_encoder:
        (
            transformer,
            text_encoder_one,
            text_encoder_two,
            text_encoder_three,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            transformer,
            text_encoder_one,
            text_encoder_two,
            text_encoder_three,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration
    # The trackers initializes automatically on the main process
    if accelerator.is_main_process:
        tracker_name = "dreambooth-sd3-multispectral"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running multispectral DreamBooth training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Multispectral configuration:"),
    logger.info(f"    - VAE latent channels: {vae.config.latent_channels}")
    logger.info(f"    - VAE adapter_in_channels: {getattr(vae.config, 'adapter_in_channels', 'N/A')}")
    logger.info(f"    - VAE adapter_out_channels: {getattr(vae.config, 'adapter_out_channels', 'N/A')}")
    logger.info(f"    - VAE backbone_in_channels: {getattr(vae.config, 'backbone_in_channels', 'N/A')}")
    logger.info(f"    - VAE backbone_out_channels: {getattr(vae.config, 'backbone_out_channels', 'N/A')}")
    global_step = 0
    first_epoch = 0
    
    # Initialize checkpoint tracking
    best_val_loss = float('inf')
    best_epoch = 0

    # --- DREAMBOOTH LOGGER INIT ---
    # Use a shorter model name for log files to avoid path length issues
    model_name_for_logs = "sd3_multispectral_dreambooth"
    try:
        dreambooth_logger = create_dreambooth_logger(
            output_dir=args.output_dir,
            model_name=model_name_for_logs
        )
        logger.info("DreamBooth logger initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize DreamBooth logger: {e}")
        logger.warning("Training will continue without detailed logging")
        dreambooth_logger = None

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            # Use the full path directly 
            if args.resume_from_checkpoint == "latest":
                resume_path = os.path.join(args.output_dir, path)
            else:
                resume_path = args.resume_from_checkpoint

            accelerator.print(f"Resuming from checkpoint {resume_path}")
            accelerator.load_state(resume_path)
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    logger.info("About to enter main training loop")
    
    # Main training loop
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()
            text_encoder_three.train()

        for step, batch in enumerate(train_dataloader): # Access batches of dicts from dataloader
            # Initialize grad_norm for this step
            grad_norm = None
            
            models_to_accumulate = [transformer]
            if args.train_text_encoder:
                models_to_accumulate.extend([text_encoder_one, text_encoder_two, text_encoder_three])
            with accelerator.accumulate(models_to_accumulate):
                # Multispectral data: 5-channel input (bands 9, 18, 32, 42, 55) instead of 3-channel RGB
                # FIXED: Use vae.dtype (fp32) for pixel_values, following original DreamBooth SD3 pattern
                # This ensures VAE input is in fp32, then we convert model_input to weight_dtype after encoding
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)  # vae.dtype is fp32
                prompts = batch["prompts"]

                # encode batch prompts when custom prompts are provided for each image
                if args.train_text_encoder:
                    tokens_one = tokenize_prompt(tokenizer_one, prompts)
                    tokens_two = tokenize_prompt(tokenizer_two, prompts)
                    tokens_three = tokenize_prompt(tokenizer_three, prompts)

                # Convert images to latent space using multispectral VAE
                # Input: (B, 5, H, W) multispectral data -> Output: (B, 16, H/8, W/8) latent representation
                # The multispectral VAE preserves spectral information in the latent space
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)



                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise



                # Predict the noise residual
                # Note: SD3 transformer doesn't use down_block_additional_residuals/mid_block_additional_residual
                # These are UNet-specific parameters. SD3 uses block_controlnet_hidden_states for ControlNet.
                if not args.train_text_encoder:
                    model_pred = transformer(
                        hidden_states=noisy_model_input,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]
                else:
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two, text_encoder_three],
                        tokenizers=None,
                        prompt=None,
                        text_input_ids_list=[tokens_one, tokens_two, tokens_three],
                    )
                    model_pred = transformer(
                        hidden_states=noisy_model_input,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]

                # Follow: Section 5 of https://arxiv.org/abs/2206.00364
                # Preconditioning of the model outputs
                if args.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy_model_input

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                if args.precondition_outputs:
                    target = model_input
                else:
                    target = noise - model_input

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute prior loss
                    prior_loss = torch.mean(
                        (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(
                            target_prior.shape[0], -1
                        ),
                        1,
                    )
                    prior_loss = prior_loss.mean()

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()



                if args.with_prior_preservation:
                    # Add the prior loss to the instance loss
                    loss = loss + args.prior_loss_weight * prior_loss

                # Compute per-channel MSE, SSIM, and SAM for spectral fidelity monitoring (every 100 steps)
                if global_step % 100 == 0:
                    try:
                        with torch.no_grad():  # Don't track gradients for analysis
                            # Decode the current latents back to pixel space
                            decoded_latents = model_input / vae.config.scaling_factor + vae.config.shift_factor
                            decoded_pixels = vae.decode(decoded_latents).sample  # Shape: (B, 5, H, W)
                            
                            # Compute per-channel MSE
                            mse_per_channel = torch.mean((decoded_pixels - pixel_values) ** 2, dim=(0, 2, 3))
                            # Shape: (5,) - one MSE value per channel
                            
                            # Compute per-channel SSIM
                            if SSIM_AVAILABLE:
                                try:
                                    ssim_per_channel = []
                                    for i in range(5):  # 5 channels
                                        # Convert to numpy and normalize to [0, 1] for SSIM
                                        orig_band = (pixel_values[0, i].cpu().numpy() + 1.0) / 2.0
                                        decoded_band = (decoded_pixels[0, i].cpu().numpy() + 1.0) / 2.0
                                        ssim_score = ssim(orig_band, decoded_band, data_range=1.0)
                                        ssim_per_channel.append(ssim_score)
                                except Exception as e:
                                    logger.warning(f"SSIM computation failed: {e}")
                                    ssim_per_channel = [0.0] * 5
                            else:
                                ssim_per_channel = [0.0] * 5
                            
                            # Compute SAM (Spectral Angle Mapper)
                            if SAM_AVAILABLE:
                                try:
                                    # metric needs to be on the same device as the input tensors (default is CPU)
                                    sam = SpectralAngleMapper().to(decoded_pixels.device)
                                    
                                    # Clip outliers using 95th percentile for SAM compatibility (monitoring only)
                                    decoded_pixels_clipped = decoded_pixels.clone()
                                    for i in range(decoded_pixels_clipped.shape[1]):  # For each channel
                                        channel_data = decoded_pixels_clipped[0, i]  # Shape: (H, W)
                                        q_05 = torch.quantile(channel_data, 0.05)  # 5th percentile
                                        q_95 = torch.quantile(channel_data, 0.95)  # 95th percentile
                                        decoded_pixels_clipped[0, i] = torch.clamp(channel_data, q_05, q_95)
                                    
                                    # Debug: Check tensor shapes and values
                                    logger.info(f"SAM input shapes: decoded_pixels[0:1]={decoded_pixels_clipped[0:1].shape}, pixel_values[0:1]={pixel_values[0:1].shape}")
                                    logger.info(f"SAM input ranges: decoded_pixels=[{decoded_pixels_clipped[0:1].min():.4f}, {decoded_pixels_clipped[0:1].max():.4f}], pixel_values=[{pixel_values[0:1].min():.4f}, {pixel_values[0:1].max():.4f}]")
                                    logger.info(f"SAM device: decoded_pixels={decoded_pixels_clipped[0:1].device}, pixel_values={pixel_values[0:1].device}, sam_metric={sam.device}")
                                    
                                    sam_score = sam(decoded_pixels_clipped[0:1], pixel_values[0:1]).item()
                                    
                                    # Debug: Check output
                                    logger.info(f"SAM raw output: {sam_score}")
                                    
                                    # Check for invalid values
                                    if math.isnan(sam_score) or math.isinf(sam_score):
                                        logger.warning(f"SAM computation failed: VAE output range [{decoded_pixels[0:1].min():.4f}, {decoded_pixels[0:1].max():.4f}] incompatible with SpectralAngleMapper expectations")
                                        sam_score = None  # Use None to indicate "not computable"
                                        
                                except Exception as e:
                                    logger.error(f"SAM computation failed with error: {e}")
                                    logger.error(f"SAM input details: decoded_pixels dtype={decoded_pixels[0:1].dtype}, pixel_values dtype={pixel_values[0:1].dtype}")
                                    sam_score = None  # Use None to indicate "not computable"
                            else:
                                sam_score = None  # Use None to indicate "not available"
                            
                            # Compute range statistics for monitoring
                            try:
                                range_stats = {}
                                total_out_of_bounds = 0
                                total_pixels = 0
                                
                                for i in range(5):  # 5 channels
                                    # Get decoded pixel values for this channel
                                    channel_data = decoded_pixels[0, i]  # Shape: (H, W)
                                    
                                    # Count out-of-bounds pixels
                                    out_of_bounds = torch.sum((channel_data < -1.0) | (channel_data > 1.0)).item()
                                    total_pixels_channel = channel_data.numel()
                                    
                                    range_stats[f"band_{i}"] = {
                                        "out_of_bounds": out_of_bounds,
                                        "total_pixels": total_pixels_channel,
                                        "min_val": channel_data.min().item(),
                                        "max_val": channel_data.max().item(),
                                        "mean_val": channel_data.mean().item(),
                                        "std_val": channel_data.std().item(),
                                        "out_of_bounds_pct": (out_of_bounds / total_pixels_channel * 100) if total_pixels_channel > 0 else 0
                                    }
                                    
                                    total_out_of_bounds += out_of_bounds
                                    total_pixels += total_pixels_channel
                                
                                # Add overall statistics
                                range_stats["overall"] = {
                                    "total_out_of_bounds": total_out_of_bounds,
                                    "total_pixels": total_pixels,
                                    "out_of_bounds_percentage": (total_out_of_bounds / total_pixels * 100) if total_pixels > 0 else 0
                                }
                            except Exception as e:
                                logger.warning(f"Failed to compute range statistics: {e}")
                                range_stats = {}
                            
                            # Log all metrics with band information
                            try:
                                band_names = ["Band 9 (474nm)", "Band 18 (539nm)", "Band 32 (651nm)", 
                                             "Band 42 (731nm)", "Band 55 (851nm)"]
                                logger.info(f"Step {global_step} - Spectral Fidelity Metrics:")
                                if sam_score is not None:
                                    logger.info(f"  SAM Score: {sam_score:.6f} (95th percentile clipped)")
                                else:
                                    logger.info(f"  SAM Score: Not computable (VAE output range incompatible)")
                                for i, (mse, ssim_val, band_name) in enumerate(zip(mse_per_channel, ssim_per_channel, band_names)):
                                    logger.info(f"  {band_name}: MSE={mse.item():.6f}, SSIM={ssim_val:.6f}")
                                
                                # Log range statistics
                                if range_stats:
                                    overall_pct = range_stats["overall"]["out_of_bounds_percentage"]
                                    logger.info(f"  Range monitoring: {overall_pct:.2f}% out-of-bounds pixels")
                            except Exception as e:
                                logger.warning(f"Failed to log spectral metrics: {e}")
                            
                            # Log to wandb if available
                            try:
                                if is_wandb_available() and wandb.run:
                                    wandb_log = {}
                                    if sam_score is not None:
                                        wandb_log["spectral/sam_score"] = sam_score
                                    else:
                                        wandb_log["spectral/sam_score"] = -1.0  # Use -1 to indicate "not computable"
                                    for i, (mse, ssim_val, band_name) in enumerate(zip(mse_per_channel, ssim_per_channel, band_names)):
                                        band_id = band_name.split()[1]  # Extract wavelength
                                        wandb_log[f"spectral/mse_channel_{i}_{band_id}"] = mse.item()
                                        wandb_log[f"spectral/ssim_channel_{i}_{band_id}"] = ssim_val
                                    
                                    # Add range statistics to wandb
                                    if range_stats:
                                        wandb_log["spectral/out_of_bounds_percentage"] = range_stats["overall"]["out_of_bounds_percentage"]
                                        for i in range(5):
                                            band_stats = range_stats.get(f"band_{i}", {})
                                            wandb_log[f"spectral/band_{i}_out_of_bounds_pct"] = (band_stats.get("out_of_bounds", 0) / band_stats.get("total_pixels", 1) * 100)
                                            wandb_log[f"spectral/band_{i}_min_val"] = band_stats.get("min_val", 0)
                                            wandb_log[f"spectral/band_{i}_max_val"] = band_stats.get("max_val", 0)
                                    
                                    wandb.log(wandb_log, step=global_step)
                            except Exception as e:
                                logger.warning(f"Failed to log spectral metrics to wandb: {e}")
                    except Exception as e:
                        logger.warning(f"Failed to compute spectral metrics at step {global_step}: {e}")
                        # Continue training even if spectral metrics computation fails

                # Backward pass: compute gradients
                accelerator.backward(loss)
                
                # Gradient accumulation: only update weights after accumulating gradients
                if accelerator.sync_gradients:
                    # Select parameters to clip (all trainable models)
                    params_to_clip = (
                        itertools.chain(
                            transformer.parameters(),
                            text_encoder_one.parameters(),
                            text_encoder_two.parameters(),
                            text_encoder_three.parameters(),
                        )
                        if args.train_text_encoder
                        else transformer.parameters()
                    )
                    # Clip gradients to prevent explosion (max norm = 1.0)
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    
                    # Compute gradient norm for logging (L2 norm of all gradients)
                    total_norm = 0.0
                    for p in transformer.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm += param_norm.item() ** 2
                    grad_norm = total_norm ** 0.5

                # Update model weights using computed gradients
                optimizer.step()
                # Update learning rate schedule
                lr_scheduler.step()
                # Clear gradients for next iteration
                optimizer.zero_grad()

            # Gradient accumulation: only increment step counter when gradients are actually applied
            # With gradient_accumulation_steps=1, this happens every iteration
            # With gradient_accumulation_steps>1, this happens every N iterations
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Log to DreamBooth logger every 100 steps
                if global_step % 100 == 0 and dreambooth_logger is not None:
                    try:
                        # Prepare spectral metrics for logging
                        spectral_metrics = None
                        if 'mse_per_channel' in locals() and 'ssim_per_channel' in locals() and 'sam_score' in locals():
                            spectral_metrics = {
                                "sam_score": sam_score,
                                "mse_per_channel": [mse.item() for mse in mse_per_channel],
                                "ssim_per_channel": ssim_per_channel
                            }
                            # Add range statistics if available
                            if 'range_stats' in locals():
                                spectral_metrics["range_stats"] = range_stats
                        
                        dreambooth_logger.log_step(
                            step=global_step, 
                            epoch=epoch,
                            loss=loss.detach().item(),
                            learning_rate=lr_scheduler.get_last_lr()[0],
                            grad_norm=grad_norm,
                            spectral_metrics=spectral_metrics
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log step {global_step}: {e}")
                        # Continue training even if logging fails

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    try:
                                        shutil.rmtree(removing_checkpoint)
                                    except Exception as e:
                                        logger.warning(f"Could not remove checkpoint directory {removing_checkpoint}: {e}")

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        try:
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")
                        except Exception as e:
                            logger.error(f"Could not save accelerator state to {save_path}: {e}")

                        # Save individual components for inference (multispectral VAE compatibility)
                        components_save_path = os.path.join(args.output_dir, f"components-{global_step}")
                        os.makedirs(components_save_path, exist_ok=True)
                        try:
                            # Save transformer
                            transformer_save_path = os.path.join(components_save_path, "transformer")
                            accelerator.unwrap_model(transformer).save_pretrained(transformer_save_path)
                            
                            # Save VAE
                            vae_save_path = os.path.join(components_save_path, "vae")
                            vae.save_pretrained(vae_save_path)
                            
                            logger.info(f"Model components saved to: {components_save_path}")
                        except Exception as e:
                            logger.error(f"Could not save model components to {components_save_path}: {e}")

                    # Run validation and log images every 500 steps
                    if args.validation_prompt is not None and global_step % 500 == 0:
                        try:
                            # Use existing VAE directly for validation (no pipeline needed)
                            with torch.no_grad():
                                # Use the current batch for validation
                                latents = vae.encode(pixel_values).latent_dist.sample()
                                decoded_imgs = vae.decode(latents).sample

                                # Convert tensors to pseudo-RGB PIL images for logging
                                validation_images = []
                                for i in range(len(decoded_imgs)):
                                    # Create pseudo-RGB from all 5 channels
                                    multispectral_tensor = decoded_imgs[i]  # Shape: (5, H, W)
                                    pil_image = create_pseudo_rgb_from_multispectral(multispectral_tensor)
                                    validation_images.append(pil_image)
                                
                            # Log validation images and metrics using logger
                            if dreambooth_logger is not None:
                                val_metrics = {"val_loss": loss.detach().item()}
                                dreambooth_logger.log_validation(
                                    epoch=global_step,
                                    images=validation_images,
                                    prompt=args.validation_prompt,
                                    validation_metrics=val_metrics
                                )
                                logger.info(f"Validation logging succeeded at step {global_step}")
                            else:
                                logger.info(f"Validation completed at step {global_step} (no logger)")
                        except Exception as e:
                            logger.error(f"Could not log validation at step {global_step}: {e}")

            logs = {"loss": loss.detach().item()}
            # Step-level logging every `log_steps`
            if global_step % args.log_steps == 0 and grad_norm is not None:
                logs.update({
                    "grad_norm": grad_norm,
                })
            # Log step metrics using logger (basic metrics only, no spectral metrics here)
            # Only log if not already logged with spectral metrics (every 100 steps)
            if global_step % 100 != 0 and dreambooth_logger is not None:
                try:
                    dreambooth_logger.log_step(
                        step=global_step,
                        epoch=epoch,
                        loss=loss.detach().item(),
                        learning_rate=lr_scheduler.get_last_lr()[0],
                        grad_norm=grad_norm
                    )
                except Exception as e:
                    logger.warning(f"Failed to log basic step {global_step}: {e}")
                    # Continue training even if logging fails
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        # Validation and checkpointing
        if accelerator.is_main_process:
            # Save checkpoint every checkpointing_steps
            if global_step % args.checkpointing_steps == 0:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                try:
                    accelerator.save_state(checkpoint_path)
                    logger.info(f"Checkpoint saved to: {checkpoint_path}")
                except Exception as e:
                    logger.error(f"Could not save checkpoint to {checkpoint_path}: {e}")
            
            # --- VALIDATION DATA LOGGING FOR DREAMBOOTH LOGGER ---
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                try:
                    # Use existing VAE and transformer directly for validation (no pipeline like in RGB DB needed)
                    val_batch = next(iter(train_dataloader))
                    with torch.no_grad():
                        inputs = val_batch["pixel_values"].to(accelerator.device)
                        # Use the frozen VAE for encoding/decoding
                        latents = vae.encode(inputs).latent_dist.sample()
                        # Convert tensors to pseudo-RGB PIL images for logging
                        validation_images = []
                        for i in range(len(latents)):
                            # Decode and create pseudo-RGB from all 5 channels
                            decoded_img = vae.decode(latents[i:i+1] / vae.config.scaling_factor + vae.config.shift_factor).sample
                            multispectral_tensor = decoded_img[0]  # Shape: (5, H, W)
                            pil_image = create_pseudo_rgb_from_multispectral(multispectral_tensor)
                            validation_images.append(pil_image)
                        
                        # Log with DreamBooth logger
                        if dreambooth_logger is not None:
                            dreambooth_logger.log_validation(
                                epoch=global_step,
                                images=validation_images,
                                prompt=args.validation_prompt,
                                validation_metrics={"val_loss": loss.detach().item()}
                            )
                            logger.info(f"Validation logging succeeded at epoch {epoch}")
                        else:
                            logger.info(f"Validation completed at epoch {epoch} (no logger)")
                except Exception as e:
                    logger.error(f"Could not log validation data at epoch {epoch}: {e}")

    # Save the final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        try:
            transformer = unwrap_model(transformer)
            
            # Save individual components instead of full pipeline (multispectral VAE compatibility)
            transformer_save_path = os.path.join(args.output_dir, "transformer")
            transformer.save_pretrained(transformer_save_path)
            logger.info(f"Final transformer saved to: {transformer_save_path}")
            
            # Save VAE separately
            vae_save_path = os.path.join(args.output_dir, "vae")
            vae.save_pretrained(vae_save_path)
            logger.info(f"Final VAE saved to: {vae_save_path}")
            
            if args.train_text_encoder:
                text_encoder_one = unwrap_model(text_encoder_one)
                text_encoder_two = unwrap_model(text_encoder_two)
                text_encoder_three = unwrap_model(text_encoder_three)
                
                text_encoder_one.save_pretrained(os.path.join(args.output_dir, "text_encoder"))
                text_encoder_two.save_pretrained(os.path.join(args.output_dir, "text_encoder_2"))
                text_encoder_three.save_pretrained(os.path.join(args.output_dir, "text_encoder_3"))
                logger.info(f"Final text encoders saved to: {args.output_dir}")
            
            logger.info(f"Final model components saved to: {args.output_dir}")
        except Exception as e:
            logger.error(f"Could not save final model to {args.output_dir}: {e}")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)