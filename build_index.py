"""
build_index.py — Build a FAISS IndexFlatL2 from known faces.

Usage:
    python build_index.py
    python build_index.py --faces-dir ./faces --output faiss.index

This script:
1. Reads all images from faces/
2. Detects + extracts ArcFace embeddings
3. Builds a FAISS IndexFlatL2
4. Saves index + name mapping to disk
"""

import os
import cv2
import faiss
import pickle
import argparse
import numpy as np
from pathlib import Path

from models import SCRFD, ArcFace


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS index from known faces")
    parser.add_argument("--det-weight",        type=str, default="./weights/det_500m.onnx")
    parser.add_argument("--rec-weight",        type=str, default="./weights/w600k_mbf.onnx")
    parser.add_argument("--faces-dir",         type=str, default="./faces")
    parser.add_argument("--output",            type=str, default="./faiss.index")
    parser.add_argument("--names-output",      type=str, default="./faiss_names.pkl")
    parser.add_argument("--confidence-thresh", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("BUILDING FAISS INDEX")
    print("=" * 50)
    print(f"  Faces dir : {args.faces_dir}")
    print(f"  Index out : {args.output}")

    print("\nLoading models...")
    detector   = SCRFD(args.det_weight, input_size=(640, 640), conf_thres=args.confidence_thresh)
    recognizer = ArcFace(args.rec_weight)

    embeddings = []
    names      = []

    print("\nProcessing faces...")
    for filename in sorted(os.listdir(args.faces_dir)):
        if Path(filename).suffix.lower() not in IMAGE_EXTS:
            continue

        name       = Path(filename).stem
        image_path = os.path.join(args.faces_dir, filename)
        image      = cv2.imread(image_path)

        if image is None:
            print(f"  [WARN] Cannot read {filename}, skipping")
            continue

        bboxes, kpss = detector.detect(image, max_num=1)
        if kpss is None or len(kpss) == 0:
            print(f"  [WARN] No face in {filename}, skipping")
            continue

        embedding = recognizer(image, kpss[0])

        # Normalize for cosine similarity via L2 distance
        embedding = embedding / np.linalg.norm(embedding)

        embeddings.append(embedding)
        names.append(name)
        print(f"  [OK] {filename} → {name}  (embedding shape: {embedding.shape})")

    if not embeddings:
        print("\nNo valid faces found. Exiting.")
        return

    # Stack into matrix
    embedding_matrix = np.vstack(embeddings).astype(np.float32)

    # Build FAISS IndexFlatL2
    dim   = embedding_matrix.shape[1]  # 512 for ArcFace
    index = faiss.IndexFlatL2(dim)
    index.add(embedding_matrix)

    # Save index and names
    faiss.write_index(index, args.output)
    with open(args.names_output, "wb") as f:
        pickle.dump(names, f)

    print(f"\n  Indexed {index.ntotal} faces")
    print(f"  Embedding dim : {dim}")
    print(f"  Index saved   : {args.output}")
    print(f"  Names saved   : {args.names_output}")
    print("\nDone. Run main.py with --use-faiss flag.")
    print("=" * 50)


if __name__ == "__main__":
    main()
