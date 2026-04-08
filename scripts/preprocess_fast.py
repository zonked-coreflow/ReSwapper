"""Fast VGGFace2-HQ preprocessing using GPU RetinaFace + CPU ArcFace.

Uses PyTorch RetinaFace on GPU for landmark detection (~76 img/s),
and ONNX ArcFace on CPU for embedding extraction. Much faster than
InsightFace's full pipeline on CPU (~1 img/s).

Usage:
    python scripts/preprocess_fast.py \
        --input_dir data/raw/VGGFace2-HQ/merged \
        --output_dir data/preprocessed_vggface2 \
        --resolution 1024
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage import transform as trans


ARCFACE_DST = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
], dtype=np.float32)


def align_face(img, kps, image_size):
    ratio = float(image_size) / 112.0
    dst = ARCFACE_DST * ratio
    if image_size != 128:
        offset = (128 / 32768) * image_size - 0.5
        dst[:, 0] += offset
        dst[:, 1] += offset
    tform = trans.SimilarityTransform()
    tform.estimate(kps, dst)
    M = tform.params[0:2, :]
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped, M


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    args = parser.parse_args()

    INPUT = Path(args.input_dir)
    OUTPUT = Path(args.output_dir)
    RES = args.resolution

    # GPU face detection (for landmarks)
    from facexlib.detection import init_detection_model
    det = init_detection_model("retinaface_resnet50", device="cuda")
    print("RetinaFace loaded on GPU", flush=True)

    # CPU ArcFace embedding via ONNX
    import onnxruntime as ort
    arcface_path = Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx"
    arcface_session = ort.InferenceSession(str(arcface_path), providers=["CPUExecutionProvider"])
    arcface_input_name = arcface_session.get_inputs()[0].name
    arcface_output_name = arcface_session.get_outputs()[0].name
    print("ArcFace loaded on CPU", flush=True)

    def get_landmarks(img):
        with torch.no_grad():
            bboxes = det.detect_faces(img, 0.5)
        if len(bboxes) == 0:
            return None
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
        best = int(np.argmax(areas))
        bbox = bboxes[best]
        if len(bbox) >= 15:
            kps = np.array([[bbox[5 + i * 2], bbox[6 + i * 2]] for i in range(5)], dtype=np.float32)
            return kps
        return None

    def get_embedding(face_img):
        face_112 = cv2.resize(face_img, (112, 112))
        face_112 = cv2.cvtColor(face_112, cv2.COLOR_BGR2RGB)
        face_112 = np.transpose(face_112, (2, 0, 1)).astype(np.float32)
        face_112 = (face_112 - 127.5) / 127.5
        face_112 = np.expand_dims(face_112, 0)
        emb = arcface_session.run([arcface_output_name], {arcface_input_name: face_112})[0][0]
        return (emb / np.linalg.norm(emb)).astype(np.float32)

    identities = sorted([d for d in INPUT.iterdir() if d.is_dir()])
    print(f"Processing {len(identities)} identities at {RES}px", flush=True)

    total = 0
    skipped = 0
    t0 = time.time()

    for i, id_dir in enumerate(identities):
        out_dir = OUTPUT / id_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in id_dir.glob("*.jpg"):
            stem = img_path.stem
            out_img = out_dir / f"{stem}.jpg"
            out_emb = out_dir / f"{stem}_emb.npy"

            if out_img.exists() and out_emb.exists():
                total += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            kps = get_landmarks(img)
            if kps is None:
                skipped += 1
                continue

            aligned, M = align_face(img, kps, RES)
            embedding = get_embedding(aligned)

            cv2.imwrite(str(out_img), aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])
            np.save(str(out_emb), embedding)
            total += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = total / max(elapsed, 1)
            eta = (len(identities) - i - 1) / ((i + 1) / max(elapsed, 1))
            print(f"[{i+1}/{len(identities)}] {total} faces, {skipped} skip, "
                  f"{rate:.1f}/s, ETA {eta/3600:.1f}h", flush=True)

    elapsed = time.time() - t0
    print(f"DONE: {total} faces in {elapsed/3600:.1f}h "
          f"({total/max(elapsed, 1):.1f}/s), {skipped} skipped", flush=True)


if __name__ == "__main__":
    main()
