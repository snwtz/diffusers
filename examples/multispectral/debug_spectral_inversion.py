import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from diffusers.models.autoencoders.autoencoder_kl_multispectral_adapter import AutoencoderKLMultispectralAdapter as AutoencoderKL
from vae_multispectral_dataloader import VAEMultispectralDataset


def debug_inversion(
    model_dir,
    val_file_list,
    output_dir,
    batch_size=1,
    num_batches=10,
    invert_recon=False
):
    """
    This script checks for spectral inversion in a trained VAE.
    We log per-pixel spectral curves (original vs reconstructed) and their correlation.

    Key checks:
    - Are reconstructed spectral signatures strongly negatively correlated with the originals?
    - Is the correlation consistent across foreground pixels?
    - Does negating the reconstruction fix the issue? (optional)

    Parameters:
    - model_dir: Path to trained VAE
    - val_file_list: Text file with validation image paths
    - output_dir: Where to save logs
    - batch_size: Batch size for evaluation
    - num_batches: Number of batches to evaluate (for quick debugging)
    - invert_recon: If True, we negate the reconstructed output before comparing
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = AutoencoderKL.from_pretrained(model_dir).to(device)
    if hasattr(model.output_adapter, "global_scale"):
        scale_val = model.output_adapter.global_scale.item()
        print(f"[INFO] Global output scale (decoder): {scale_val:.4f}")
        if scale_val < 0:
            print("[WARNING] global_scale is negative. This may cause spectral inversion.")

    model.eval()

    # Load dataset
    dataset = VAEMultispectralDataset(
        file_list_path=val_file_list,
        resolution=512,
        use_cache=False,
        return_mask=True
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_corrs = []

    with torch.no_grad():
        for idx, (x, mask) in enumerate(tqdm(loader, desc="Debugging inversion")):
            if idx >= num_batches:
                break

            x = x.to(device)
            mask = mask.to(device)

            recon, _ = model(x)
            if hasattr(recon, "sample"):
                recon = recon.sample

            if invert_recon:
                recon = -1 * recon

            # Focus on a random pixel from the leaf region (mask == 1)
            m = mask[0, 0].cpu().numpy()
            ys, xs = np.where(m > 0)
            if len(xs) == 0:
                print(f"No leaf region in sample {idx}, skipping")
                continue

            yx = np.random.choice(len(xs))
            y, x_ = ys[yx], xs[yx]
            orig_spec = x[0, :, y, x_].cpu().numpy()
            recon_spec = recon[0, :, y, x_].cpu().numpy()

            # --- DEBUG: Checking sign-flip correlation to diagnose spectral inversion ---
            recon_spec_inverted = -1 * recon_spec
            corr_inverted = np.corrcoef(orig_spec, recon_spec_inverted)[0, 1]
            # -------------------------------------------------------------

            corr = np.corrcoef(orig_spec, recon_spec)[0, 1]
            all_corrs.append(corr)

            print(f"Sample {idx} | pixel ({y}, {x_})")
            print("  Original spectrum:", np.round(orig_spec, 3))
            print("  Reconstructed spectrum:", np.round(recon_spec, 3))
            print(f"  Correlation: {corr:.3f} {'[INVERTED]' if corr < -0.5 else ''}")
            print(f"  Corr (inverted recon): {corr_inverted:.3f}\n")

    avg_corr = np.mean(all_corrs)
    print("================================================")
    print(f"[Summary] Average correlation across {len(all_corrs)} pixels: {avg_corr:.3f}")
    print("Note: if corr_inverted >> corr, inversion is a likely cause.")
    print("Interpretation:")
    print("- Close to 1.0 => Good match")
    print("- Close to -1.0 => Likely inversion")
    print("- Around 0 => Unrelated signal")
    print("================================================")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--val_file_list", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--invert_recon", action="store_true")
    args = parser.parse_args()

    debug_inversion(
        model_dir=args.model_dir,
        val_file_list=args.val_file_list,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        invert_recon=args.invert_recon
    )