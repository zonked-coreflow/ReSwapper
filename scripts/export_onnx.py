"""Export ReSwapper v2 generator to ONNX for deployment.

Usage:
    python scripts/export_onnx.py \
        --checkpoint checkpoints/checkpoint-500000.pt \
        --output reswapper_v2.onnx \
        --resolution 512
"""

import argparse

import torch

from reswapper.config import ReSwapperConfig
from reswapper.models.generator import FaceSwapGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Export generator to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="reswapper_v2.onnx")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main():
    args = parse_args()
    config = ReSwapperConfig.from_yaml(args.config)

    # Build generator
    generator = FaceSwapGenerator(
        latent_channels=config.generator.latent_channels,
        base_channels=config.generator.base_channels,
        bottleneck_channels=config.generator.bottleneck_channels,
        id_dim=config.identity.embedding_dim,
        num_id_tokens=config.identity.num_tokens,
        parsing_channels=config.generator.parsing_channels,
    )

    # Load weights
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.use_ema and "ema" in checkpoint:
        generator.load_state_dict(checkpoint["ema"])
        print("Loaded EMA weights")
    else:
        generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    # Create dummy inputs
    latent_size = args.resolution // 8  # FLUX VAE 8x compression
    target_latent = torch.randn(1, config.generator.latent_channels, latent_size, latent_size)
    id_embedding = torch.randn(1, config.identity.embedding_dim)
    parsing_map = torch.randn(1, config.generator.parsing_channels, latent_size, latent_size)

    # Export
    print(f"Exporting to {args.output} (opset {args.opset})...")
    torch.onnx.export(
        generator,
        (target_latent, id_embedding, parsing_map),
        args.output,
        opset_version=args.opset,
        input_names=["target_latent", "id_embedding", "parsing_map"],
        output_names=["swapped_latent", "swap_mask"],
        dynamic_axes={
            "target_latent": {0: "batch", 2: "height", 3: "width"},
            "parsing_map": {0: "batch", 2: "height", 3: "width"},
            "id_embedding": {0: "batch"},
            "swapped_latent": {0: "batch", 2: "height", 3: "width"},
            "swap_mask": {0: "batch", 2: "height", 3: "width"},
        },
    )

    # Verify
    import onnx
    model = onnx.load(args.output)
    onnx.checker.check_model(model)
    print(f"ONNX model saved and verified: {args.output}")

    # Print size
    import os
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
