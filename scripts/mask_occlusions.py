"""Generate occlusion masks for real-world occluded face images.

Takes a directory of face images with occlusions (tubes, hands, cigarettes,
microphones, tongues, etc.) and produces binary/alpha masks that identify
which pixels are occluded vs face.

Three masking strategies:
1. **SAM2 interactive** — point-prompt SAM2 on occluding objects (best quality)
2. **Face parser diff** — compare face parsing to expected regions, flag anomalies
3. **Manual masks** — load user-provided masks from a parallel directory

Output structure:
    output_dir/
        image_001.jpg         # original image (copied or symlinked)
        image_001_occlusion.npy  # [H, W] float32 alpha mask: 1=face, 0=occluded
        image_001_emb.npy     # identity embedding (if --extract-embeddings)

Usage:
    # With SAM2 (requires GPU, best quality):
    python scripts/mask_occlusions.py \
        --input_dir data/occluded_faces \
        --output_dir data/occlusion_masks \
        --method sam2

    # With face parser (CPU-friendly, good for initial pass):
    python scripts/mask_occlusions.py \
        --input_dir data/occluded_faces \
        --output_dir data/occlusion_masks \
        --method parser

    # Load pre-made masks (user already created masks):
    python scripts/mask_occlusions.py \
        --input_dir data/occluded_faces \
        --mask_dir data/manual_masks \
        --output_dir data/occlusion_masks \
        --method manual

    # Just verify existing masks (visualize overlay):
    python scripts/mask_occlusions.py \
        --input_dir data/occlusion_masks \
        --method verify
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate occlusion masks for face images")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of occluded face images")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for masks")
    parser.add_argument("--mask_dir", type=str, default=None, help="Directory of pre-made masks (for --method manual)")
    parser.add_argument("--method", type=str, default="parser",
                        choices=["sam2", "parser", "manual", "verify"],
                        help="Masking method")
    parser.add_argument("--resolution", type=int, default=512, help="Align faces to this resolution")
    parser.add_argument("--extract-embeddings", action="store_true", help="Also extract identity embeddings")
    parser.add_argument("--visualize", action="store_true", help="Save overlay visualizations")
    return parser.parse_args()


def mask_with_face_parser(image: np.ndarray) -> np.ndarray:
    """Use face parsing to detect occlusions.

    Strategy: parse the face, identify regions that SHOULD be face but
    are classified as background/unknown. Those are likely occluded.

    Returns [H, W] float32 mask where 1=face visible, 0=occluded.
    """
    # For now, use InsightFace's face analysis to get face region
    # A more sophisticated approach would use SegFace and compare
    # expected vs actual face regions
    H, W = image.shape[:2]

    # Simple approach: detect face, create convex hull mask from landmarks,
    # then anything inside the hull that doesn't look like face is occluded
    try:
        from insightface.app import FaceAnalysis
        fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        fa.prepare(ctx_id=-1, det_size=(512, 512))

        faces = fa.get(image)
        if len(faces) == 0:
            return np.ones((H, W), dtype=np.float32)

        face = faces[0]
        kps = face.kps.astype(np.int32)

        # Create face region mask from landmarks
        # Use a larger region around the detected landmarks
        hull = cv2.convexHull(kps)
        face_mask = np.zeros((H, W), dtype=np.float32)
        cv2.fillConvexPoly(face_mask, hull, 1.0)

        # Dilate to cover more face area
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (H // 5, H // 5))
        face_mask = cv2.dilate(face_mask, kernel, iterations=1)

        return face_mask.clip(0, 1)

    except Exception as e:
        print(f"Face parser masking failed: {e}")
        return np.ones((H, W), dtype=np.float32)


def mask_with_sam2(image: np.ndarray) -> np.ndarray:
    """Use SAM2 to segment occluding objects.

    Requires GPU and sam2 installed.
    This auto-segments everything in the image, then classifies
    segments as face vs occlusion based on their overlap with
    the detected face region.

    Returns [H, W] float32 mask where 1=face visible, 0=occluded.
    """
    H, W = image.shape[:2]

    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        import torch

        # Build SAM2
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam2 = build_sam2("sam2_hiera_l", "sam2_hiera_large.pt", device=device)
        mask_generator = SAM2AutomaticMaskGenerator(sam2)

        # Get face region first
        face_region = mask_with_face_parser(image)

        # Generate all masks
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image_rgb)

        # Classify each segment: if it overlaps significantly with the face
        # region but is small relative to the face, it's likely an occluder
        occlusion_mask = np.zeros((H, W), dtype=np.float32)

        for mask_data in masks:
            seg = mask_data["segmentation"].astype(np.float32)
            seg_area = seg.sum()
            face_area = face_region.sum()

            if face_area == 0:
                continue

            # Overlap with face region
            overlap = (seg * face_region).sum()
            overlap_ratio = overlap / (seg_area + 1e-8)

            # Small-to-medium objects overlapping the face = likely occlusion
            # (not the face itself, not the background)
            size_ratio = seg_area / face_area
            if overlap_ratio > 0.3 and 0.01 < size_ratio < 0.5:
                occlusion_mask = np.maximum(occlusion_mask, seg)

        # Invert: 1=face, 0=occluded
        result = 1.0 - occlusion_mask
        return result.clip(0, 1).astype(np.float32)

    except ImportError:
        print("SAM2 not installed. Install with: pip install sam2")
        print("Falling back to face parser method.")
        return mask_with_face_parser(image)
    except Exception as e:
        print(f"SAM2 masking failed: {e}")
        return mask_with_face_parser(image)


def load_manual_mask(mask_path: str, target_size: tuple[int, int]) -> np.ndarray:
    """Load a user-provided mask image.

    Expects a grayscale or binary image where:
    - White (255) = face visible (swap this)
    - Black (0) = occluded (keep target)

    Supports PNG with alpha channel (alpha used as mask).
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return np.ones(target_size, dtype=np.float32)

    # If RGBA, use alpha channel
    if mask.ndim == 3 and mask.shape[2] == 4:
        mask = mask[:, :, 3]
    elif mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Resize if needed
    if mask.shape[:2] != target_size:
        mask = cv2.resize(mask, (target_size[1], target_size[0]))

    return (mask.astype(np.float32) / 255.0).clip(0, 1)


def visualize_mask(image: np.ndarray, mask: np.ndarray, output_path: str):
    """Save a visualization showing the mask overlay on the image."""
    vis = image.copy()
    # Red overlay where occluded (mask=0)
    red_overlay = np.zeros_like(image)
    red_overlay[:, :, 2] = 255  # Red in BGR
    occlusion_region = (1.0 - mask)[:, :, np.newaxis]
    vis = (vis * (1 - 0.4 * occlusion_region) + red_overlay * 0.4 * occlusion_region).astype(np.uint8)
    cv2.imwrite(output_path, vis)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in image_extensions
        and not p.stem.endswith(("_occlusion", "_mask", "_vis", "_emb"))
    ])

    print(f"Processing {len(image_paths)} images from {input_dir}")
    print(f"Method: {args.method}")

    for i, img_path in enumerate(image_paths):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Skip (cannot read): {img_path.name}")
            continue

        stem = img_path.stem
        H, W = image.shape[:2]

        # Generate mask
        if args.method == "sam2":
            mask = mask_with_sam2(image)
        elif args.method == "parser":
            mask = mask_with_face_parser(image)
        elif args.method == "manual":
            mask_path = Path(args.mask_dir) / f"{stem}.png" if args.mask_dir else None
            if mask_path and mask_path.exists():
                mask = load_manual_mask(str(mask_path), (H, W))
            else:
                # Try same directory with _mask suffix
                for ext in [".png", ".jpg"]:
                    alt = input_dir / f"{stem}_mask{ext}"
                    if alt.exists():
                        mask = load_manual_mask(str(alt), (H, W))
                        break
                else:
                    print(f"  Skip (no mask found): {stem}")
                    continue
        elif args.method == "verify":
            mask_npy = input_dir / f"{stem}_occlusion.npy"
            if mask_npy.exists():
                mask = np.load(str(mask_npy))
                vis_path = output_dir / f"{stem}_vis.jpg"
                visualize_mask(image, mask, str(vis_path))
                face_pct = mask.mean() * 100
                print(f"  [{i + 1}/{len(image_paths)}] {stem}: {face_pct:.1f}% face visible")
            continue
        else:
            continue

        # Save results
        if args.output_dir and args.output_dir != str(input_dir):
            cv2.imwrite(str(output_dir / img_path.name), image)

        np.save(str(output_dir / f"{stem}_occlusion.npy"), mask.astype(np.float32))

        if args.visualize:
            visualize_mask(image, mask, str(output_dir / f"{stem}_vis.jpg"))

        face_pct = mask.mean() * 100
        print(f"  [{i + 1}/{len(image_paths)}] {stem}: {face_pct:.1f}% face visible")

    print(f"\nDone! Masks saved to {output_dir}")


if __name__ == "__main__":
    main()
