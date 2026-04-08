"""Offline preprocessing for FFHQ dataset.

Same as preprocess_vggface2.py but handles FFHQ's flat directory structure
(no identity grouping). All images go into a single output directory.

Usage:
    python scripts/preprocess_ffhq.py \
        --input_dir data/raw/ffhq \
        --output_dir data/preprocessed_ffhq \
        --resolution 512
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess FFHQ for training")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--det_size", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import sys
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import face_align

    from insightface.app import FaceAnalysis
    fa = FaceAnalysis(name="buffalo_l")
    fa.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    image_paths = sorted(
        list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    )
    print(f"Processing {len(image_paths)} FFHQ images")

    processed = 0
    for i, img_path in enumerate(image_paths):
        stem = img_path.stem
        out_path = output_dir / f"{stem}.jpg"

        if out_path.exists():
            processed += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        faces = fa.get(img)
        if len(faces) == 0:
            continue

        face = faces[0]
        aligned, M = face_align.norm_crop2(img, face.kps, args.resolution)
        cv2.imwrite(str(out_path), aligned)

        embedding = face.normed_embedding
        np.save(str(output_dir / f"{stem}_emb.npy"), embedding.astype(np.float32))

        processed += 1
        if (i + 1) % 1000 == 0:
            print(f"  [{i + 1}/{len(image_paths)}] processed: {processed}")

    print(f"Done! Processed {processed}/{len(image_paths)} images to {output_dir}")


if __name__ == "__main__":
    main()
