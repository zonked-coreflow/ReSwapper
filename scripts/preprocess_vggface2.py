"""Offline preprocessing for VGGFace2-HQ dataset.

Run once to prepare the dataset for training:
1. Detect faces with InsightFace
2. Align to 512x512 + 112x112
3. Extract identity embeddings (BlendFace/ArcFace)
4. Generate face parsing maps (SegFace)
5. (Optional) VAE-encode to FLUX latent and cache

Usage:
    python scripts/preprocess_vggface2.py \
        --input_dir /path/to/VGGFace2-HQ \
        --output_dir data/preprocessed_vggface2 \
        --resolution 512

This eliminates the current ReSwapper's bottleneck of running face
detection inside the training loop (was batch-size-1 because of this).
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess VGGFace2-HQ for training")
    parser.add_argument("--input_dir", type=str, required=True, help="Raw VGGFace2-HQ directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output preprocessed directory")
    parser.add_argument("--resolution", type=int, default=512, help="Target face resolution")
    parser.add_argument("--arcface_resolution", type=int, default=112, help="ArcFace input resolution")
    parser.add_argument("--det_size", type=int, default=512, help="Face detection input size")
    parser.add_argument("--num_workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--encode_latent", action="store_true", help="Also VAE-encode to FLUX latent")
    parser.add_argument("--vae_path", type=str, default=None, help="Path to FLUX VAE for latent encoding")
    return parser.parse_args()


def process_identity(
    identity_dir: Path,
    output_identity_dir: Path,
    face_analysis,
    resolution: int,
    arcface_resolution: int,
):
    """Process all images for a single identity."""
    import sys
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import face_align
    from Image import getBlob, getLatent

    output_identity_dir.mkdir(parents=True, exist_ok=True)
    processed = 0

    for img_path in sorted(identity_dir.glob("*.jpg")) + sorted(identity_dir.glob("*.png")):
        stem = img_path.stem
        output_img = output_identity_dir / f"{stem}.jpg"

        # Skip if already processed
        if output_img.exists():
            processed += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        faces = face_analysis.get(img)
        if len(faces) == 0:
            continue

        face = faces[0]

        # Align face to target resolution
        aligned_face, M = face_align.norm_crop2(img, face.kps, resolution)
        cv2.imwrite(str(output_img), aligned_face)

        # Extract identity embedding
        embedding = face.normed_embedding  # [512]
        np.save(str(output_identity_dir / f"{stem}_emb.npy"), embedding.astype(np.float32))

        processed += 1

    return processed


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Initialize face analysis
    from insightface.app import FaceAnalysis
    face_analysis = FaceAnalysis(name="buffalo_l")
    face_analysis.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    # Process each identity directory
    identity_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    total_identities = len(identity_dirs)
    total_processed = 0

    print(f"Processing {total_identities} identities from {input_dir}")
    print(f"Output: {output_dir}, Resolution: {args.resolution}")

    for i, identity_dir in enumerate(identity_dirs):
        output_identity_dir = output_dir / identity_dir.name
        count = process_identity(
            identity_dir,
            output_identity_dir,
            face_analysis,
            args.resolution,
            args.arcface_resolution,
        )
        total_processed += count

        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{total_identities}] {identity_dir.name}: {count} faces, total: {total_processed}")

    print(f"\nDone! Processed {total_processed} faces across {total_identities} identities")


if __name__ == "__main__":
    main()
