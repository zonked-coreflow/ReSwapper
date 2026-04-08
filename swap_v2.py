"""ReSwapper v2 inference entry point.

Usage:
    python swap_v2.py \
        --target input/target.jpg \
        --source input/source.jpg \
        --output output/swapped.jpg \
        --checkpoint checkpoints/checkpoint-500000.pt \
        --resolution 512
"""

import argparse

import cv2
import torch

from reswapper.config import ReSwapperConfig
from reswapper.models.generator import FaceSwapGenerator
from reswapper.models.vae import FluxVAE
from reswapper.models.identity_encoder import IdentityEncoder
from reswapper.models.face_parser import FaceParser
from reswapper.models.ema import EMAModel
from reswapper.inference.swap import FaceSwapper


def parse_args():
    parser = argparse.ArgumentParser(description="ReSwapper v2 face swap")
    parser.add_argument("--target", type=str, required=True, help="Target image path")
    parser.add_argument("--source", type=str, required=True, help="Source image path")
    parser.add_argument("--output", type=str, required=True, help="Output image path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    parser.add_argument("--resolution", type=int, default=512, help="Face crop resolution")
    parser.add_argument("--no-paste-back", action="store_true", help="Don't paste back into original")
    parser.add_argument("--use-ema", action="store_true", default=True, help="Use EMA weights")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    config = ReSwapperConfig.from_yaml(args.config)

    # Build generator
    generator = FaceSwapGenerator(
        latent_channels=config.generator.latent_channels,
        base_channels=config.generator.base_channels,
        bottleneck_channels=config.generator.bottleneck_channels,
        id_dim=config.identity.embedding_dim,
        num_id_tokens=config.identity.num_tokens,
        parsing_channels=config.generator.parsing_channels,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if args.use_ema and "ema" in checkpoint:
        generator.load_state_dict(checkpoint["ema"])
        print("Loaded EMA weights")
    else:
        generator.load_state_dict(checkpoint["generator"])
        print("Loaded generator weights")
    generator.eval()

    # Load frozen models
    vae = FluxVAE()
    vae.load(device)

    identity_encoder = IdentityEncoder(
        model=config.identity.model,
        weights_path=config.identity.weights_path,
    )
    identity_encoder.load(device)

    face_parser = FaceParser()
    face_parser.load(device)

    # Create swapper
    swapper = FaceSwapper(
        generator=generator,
        vae=vae,
        identity_encoder=identity_encoder,
        face_parser=face_parser,
        device=device,
        resolution=args.resolution,
    )

    # Run swap
    target_image = cv2.imread(args.target)
    source_image = cv2.imread(args.source)

    if target_image is None:
        print(f"Error: cannot read {args.target}")
        return
    if source_image is None:
        print(f"Error: cannot read {args.source}")
        return

    result = swapper.swap(
        target_image,
        source_image,
        paste_back=not args.no_paste_back,
    )

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, result)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
