"""Ultra-fast VGGFace2-HQ preprocessing — everything on GPU.

VGGFace2-HQ images are already face-cropped at 512x512. We:
1. GPU RetinaFace for 5-point landmarks (for alignment)
2. GPU PyTorch ArcFace (iresnet50) for embeddings (bypasses ONNX CPU bottleneck)
3. Align + resize to target resolution

Usage:
    python scripts/preprocess_allgpu.py \
        --input_dir data/raw/VGGFace2-HQ/merged \
        --output_dir data/preprocessed_vggface2 \
        --resolution 1024 \
        --arcface_pth arcface_r50.pth
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
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
    parser.add_argument("--arcface_pth", type=str, default="arcface_r50.pth")
    args = parser.parse_args()

    INPUT = Path(args.input_dir)
    OUTPUT = Path(args.output_dir)
    RES = args.resolution
    device = torch.device("cuda")

    # GPU face detection (for landmarks)
    from facexlib.detection import init_detection_model
    det = init_detection_model("retinaface_resnet50", device="cuda")
    print("RetinaFace loaded on GPU", flush=True)

    # GPU ArcFace (PyTorch, not ONNX)
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from iresnet import iresnet50

    arcface = iresnet50().to(device)
    if Path(args.arcface_pth).exists():
        arcface.load_state_dict(torch.load(args.arcface_pth, map_location=device))
        print(f"ArcFace loaded from {args.arcface_pth} on GPU", flush=True)
    else:
        # Convert from ONNX on the fly
        print("Converting ArcFace ONNX -> PyTorch...", flush=True)
        from weight_transfer import arcface_onnx_to_pth
        onnx_path = Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx"
        arcface_onnx_to_pth(str(onnx_path), args.arcface_pth)
        arcface.load_state_dict(torch.load(args.arcface_pth, map_location=device))
        print(f"ArcFace converted and loaded on GPU", flush=True)

    arcface.eval()

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

    @torch.no_grad()
    def get_embedding_gpu(face_img):
        """Get ArcFace embedding on GPU. Input: BGR numpy, any size."""
        face_112 = cv2.resize(face_img, (112, 112))
        face_112 = cv2.cvtColor(face_112, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(face_112).permute(2, 0, 1).float().unsqueeze(0).to(device)
        tensor = (tensor - 127.5) / 127.5
        emb = arcface(tensor)
        emb = F.normalize(emb, dim=1)
        return emb.cpu().numpy()[0].astype(np.float32)

    identities = sorted([d for d in INPUT.iterdir() if d.is_dir()])
    print(f"Processing {len(identities)} identities at {RES}px (ALL GPU)", flush=True)

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
            embedding = get_embedding_gpu(aligned)

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
