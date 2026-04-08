"""Download training datasets from HuggingFace.

Datasets:
- VGGFace2-HQ: https://huggingface.co/datasets/RichardErkhov/VGGFace2-HQ
- FFHQ: https://huggingface.co/datasets/marcosv/ffhq-dataset

Usage:
    python scripts/download_datasets.py \
        --output_dir data/raw \
        --datasets vggface2 ffhq
"""

import argparse
from pathlib import Path


def download_vggface2(output_dir: Path):
    """Download VGGFace2-HQ from HuggingFace."""
    vgg_dir = output_dir / "VGGFace2-HQ"
    if vgg_dir.exists() and any(vgg_dir.iterdir()):
        print(f"VGGFace2-HQ already exists at {vgg_dir}")
        return

    print("Downloading VGGFace2-HQ (this may take a while)...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            "RichardErkhov/VGGFace2-HQ",
            repo_type="dataset",
            local_dir=str(vgg_dir),
        )
        print(f"VGGFace2-HQ saved to {vgg_dir}")
    except Exception as e:
        print(f"Failed: {e}")
        print("Manual download: https://huggingface.co/datasets/RichardErkhov/VGGFace2-HQ")


def download_ffhq(output_dir: Path):
    """Download FFHQ from HuggingFace."""
    ffhq_dir = output_dir / "ffhq"
    if ffhq_dir.exists() and any(ffhq_dir.iterdir()):
        print(f"FFHQ already exists at {ffhq_dir}")
        return

    print("Downloading FFHQ dataset...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            "marcosv/ffhq-dataset",
            repo_type="dataset",
            local_dir=str(ffhq_dir),
        )
        print(f"FFHQ saved to {ffhq_dir}")
    except Exception as e:
        print(f"Failed: {e}")
        print("Manual download: https://huggingface.co/datasets/marcosv/ffhq-dataset")


def main():
    parser = argparse.ArgumentParser(description="Download training datasets")
    parser.add_argument("--output_dir", type=str, default="data/raw")
    parser.add_argument("--datasets", nargs="+", default=["vggface2", "ffhq"],
                        choices=["vggface2", "ffhq"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "vggface2" in args.datasets:
        download_vggface2(output_dir)
    if "ffhq" in args.datasets:
        download_ffhq(output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
