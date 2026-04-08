"""Download pretrained models required for ReSwapper v2.

Models:
- FLUX VAE (from Z-Image-Turbo or FLUX.1-schnell)
- BlendFace identity encoder
- SegFace face parser
- DINOv2-B (auto-downloaded by torch.hub)

Usage:
    python scripts/download_models.py --output_dir pretrained/
"""

import argparse
import os
from pathlib import Path


def download_flux_vae(output_dir: Path):
    """Download FLUX VAE from HuggingFace."""
    vae_dir = output_dir / "flux_vae"
    if (vae_dir / "config.json").exists():
        print("FLUX VAE already downloaded")
        return

    print("Downloading FLUX VAE...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            "black-forest-labs/FLUX.1-schnell",
            local_dir=str(vae_dir),
            allow_patterns=["vae/*"],
        )
        print(f"FLUX VAE saved to {vae_dir}")
    except Exception as e:
        print(f"Failed to download FLUX VAE: {e}")
        print("Alternative: download manually from https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/main/vae")


def download_blendface(output_dir: Path):
    """Download BlendFace identity encoder."""
    bf_dir = output_dir / "blendface"
    bf_dir.mkdir(parents=True, exist_ok=True)
    weights_path = bf_dir / "blendface.pth"

    if weights_path.exists():
        print("BlendFace already downloaded")
        return

    print("Downloading BlendFace...")
    print("Note: BlendFace weights must be downloaded from https://github.com/mapooon/BlendFace")
    print(f"Place the .pth file at: {weights_path}")
    print("Falling back to ArcFace (buffalo_l) for now — it will auto-download via InsightFace.")


def download_segface(output_dir: Path):
    """Download SegFace face parser."""
    sf_dir = output_dir / "segface"
    sf_dir.mkdir(parents=True, exist_ok=True)

    if any(sf_dir.glob("*.pth")) or any(sf_dir.glob("*.pt")):
        print("SegFace already downloaded")
        return

    print("SegFace must be downloaded from the official repo:")
    print("  https://github.com/zhiqinzhu123/SegFace")
    print(f"Place weights at: {sf_dir}/")
    print("Using placeholder face parser for now.")


def download_dinov2(output_dir: Path):
    """Pre-download DINOv2-B via torch.hub."""
    print("Pre-downloading DINOv2-B backbone...")
    try:
        import torch
        torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        print("DINOv2-B cached in torch hub cache")
    except Exception as e:
        print(f"DINOv2 download failed (will retry at training time): {e}")


def main():
    parser = argparse.ArgumentParser(description="Download pretrained models")
    parser.add_argument("--output_dir", type=str, default="pretrained")
    parser.add_argument("--skip-dinov2", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    download_flux_vae(output_dir)
    download_blendface(output_dir)
    download_segface(output_dir)
    if not args.skip_dinov2:
        download_dinov2(output_dir)

    print("\nDone! Model status:")
    print(f"  FLUX VAE:  {'✓' if (output_dir / 'flux_vae' / 'vae').exists() else '✗ manual download needed'}")
    print(f"  BlendFace: {'✓' if (output_dir / 'blendface' / 'blendface.pth').exists() else '✗ manual download needed'}")
    print(f"  SegFace:   {'✓' if any((output_dir / 'segface').glob('*.pth')) else '✗ manual download needed (using placeholder)'}")
    print(f"  DINOv2-B:  auto-cached via torch.hub")


if __name__ == "__main__":
    main()
