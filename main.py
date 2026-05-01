import os
import cv2
import time
import random
import pickle
import warnings
import argparse
import logging
import numpy as np

import faiss
import onnxruntime
from typing import Union, List, Tuple
from models import SCRFD, ArcFace
from utils.helpers import compute_similarity, draw_bbox_info, draw_bbox

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Face Detection-and-Recognition")
    parser.add_argument(
        "--det-weight",
        type=str,
        default="./weights/det_10g.onnx",
        help="Path to detection model"
    )
    parser.add_argument(
        "--rec-weight",
        type=str,
        default="./weights/w600k_r50.onnx",
        help="Path to recognition model"
    )
    parser.add_argument(
        "--similarity-thresh",
        type=float,
        default=0.4,
        help="Similarity threshold between faces"
    )
    parser.add_argument(
        "--confidence-thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for face detection"
    )
    parser.add_argument(
        "--faces-dir",
        type=str,
        default="./faces",
        help="Path to faces stored dir"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="./assets/855564-hd_1920_1080_24fps.mp4",
        help="Video file or video camera source. i.e 0 - webcam"
    )
    parser.add_argument(
        "--max-num",
        type=int,
        default=0,
        help="Maximum number of face detections from a frame"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--use-faiss",
        action="store_true",
        help="Use FAISS index for fast similarity search"
    )
    parser.add_argument(
        "--faiss-index",
        type=str,
        default="./faiss.index",
        help="Path to FAISS index file"
    )
    parser.add_argument(
        "--faiss-names",
        type=str,
        default="./faiss_names.pkl",
        help="Path to FAISS names mapping file"
    )

    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), None),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def build_targets(detector, recognizer, params: argparse.Namespace) -> List[Tuple[np.ndarray, str]]:
    """
    Build targets using face detection and recognition.

    Args:
        detector (SCRFD): Face detector model.
        recognizer (ArcFaceONNX): Face recognizer model.
        params (argparse.Namespace): Command line arguments.

    Returns:
        List[Tuple[np.ndarray, str]]: A list of tuples containing feature vectors and corresponding image names.
    """
    targets = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for filename in os.listdir(params.faces_dir):
        if os.path.splitext(filename)[1].lower() not in valid_exts:
            continue
        name = os.path.splitext(filename)[0]
        image_path = os.path.join(params.faces_dir, filename)

        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Could not read {image_path}. Skipping...")
            continue

        bboxes, kpss = detector.detect(image, max_num=1)

        if kpss is None or len(kpss) == 0:
            logging.warning(f"No face detected in {image_path}. Skipping...")
            continue

        embedding = recognizer(image, kpss[0])
        targets.append((embedding, name))

    return targets


def load_faiss_index(index_path: str, names_path: str):
    """Load FAISS index and name mapping from disk."""
    index = faiss.read_index(index_path)
    with open(names_path, "rb") as f:
        names = pickle.load(f)
    logging.info(f"Loaded FAISS index with {index.ntotal} faces: {names}")
    return index, names


def faiss_identify(embedding: np.ndarray, index, names: list, similarity_thresh: float):
    """Identify a face using FAISS IndexFlatL2."""
    # Normalize embedding for cosine similarity
    emb = embedding / np.linalg.norm(embedding)
    emb = emb.reshape(1, -1).astype(np.float32)

    # Search — returns L2 distances and indices
    distances, indices = index.search(emb, k=1)
    distance = distances[0][0]
    idx      = indices[0][0]

    # Convert L2 distance to cosine similarity: sim = 1 - (dist / 2)
    similarity = 1.0 - (distance / 2.0)

    if similarity > similarity_thresh and idx >= 0:
        return names[idx], similarity
    return "Unknown", similarity


def frame_processor(
    frame: np.ndarray,
    detector: SCRFD,
    recognizer: ArcFace,
    targets: List[Tuple[np.ndarray, str]],
    colors: dict,
    params: argparse.Namespace,
    faiss_index=None,
    faiss_names=None,
) -> np.ndarray:
    """
    Process a video frame for face detection and recognition.

    Args:
        frame (np.ndarray): The video frame.
        detector (SCRFD): Face detector model.
        recognizer (ArcFace): Face recognizer model.
        targets (List[Tuple[np.ndarray, str]]): List of target feature vectors and names.
        colors (dict): Dictionary of colors for drawing bounding boxes.
        params (argparse.Namespace): Command line arguments.

    Returns:
        np.ndarray: The processed video frame.
    """
    bboxes, kpss = detector.detect(frame, params.max_num)

    for bbox, kps in zip(bboxes, kpss):
        *bbox, conf_score = bbox.astype(np.int32)
        embedding = recognizer(frame, kps)

        if params.use_faiss and faiss_index is not None:
            # FAISS fast search
            best_match_name, max_similarity = faiss_identify(
                embedding, faiss_index, faiss_names, params.similarity_thresh
            )
        else:
            # Original linear search
            max_similarity  = 0
            best_match_name = "Unknown"
            for target, name in targets:
                similarity = compute_similarity(target, embedding)
                if similarity > max_similarity and similarity > params.similarity_thresh:
                    max_similarity  = similarity
                    best_match_name = name

        if best_match_name != "Unknown":
            color = colors[best_match_name]
            draw_bbox_info(frame, bbox, similarity=max_similarity, name=best_match_name, color=color)
        else:
            draw_bbox(frame, bbox, (255, 0, 0))

    return frame


def main(params):
    setup_logging(params.log_level)

    detector = SCRFD(params.det_weight, input_size=(640, 640), conf_thres=params.confidence_thresh)
    recognizer = ArcFace(params.rec_weight)

    targets = build_targets(detector, recognizer, params)
    colors  = {name: (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)) for _, name in targets}

    # Load FAISS index if requested
    faiss_index = None
    faiss_names = None
    if params.use_faiss:
        if os.path.exists(params.faiss_index) and os.path.exists(params.faiss_names):
            faiss_index, faiss_names = load_faiss_index(params.faiss_index, params.faiss_names)
            # Add any new names from targets not yet in FAISS colors
            for name in faiss_names:
                if name not in colors:
                    colors[name] = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
            logging.info("Using FAISS for fast similarity search")
        else:
            logging.warning("FAISS index not found. Run build_index.py first. Falling back to linear search.")
            params.use_faiss = False

    cap = cv2.VideoCapture(params.source)


    if not cap.isOpened():
        raise Exception("Could not open video or webcam")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter("855564-hd_1920_1080_24fps.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame_processor(frame, detector, recognizer, targets, colors, params, faiss_index, faiss_names)

        curr_time = time.time()
        inference_fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {inference_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    if args.source.isdigit():
        args.source = int(args.source)
    main(args)
