import cv2
import time
import argparse
import numpy as np
from collections import defaultdict

from models import SCRFD, ArcFace
from utils.helpers import compute_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Video real-time performance benchmark")
    parser.add_argument("--det-weight",        type=str,   default="./weights/det_500m.onnx")
    parser.add_argument("--rec-weight",        type=str,   default="./weights/w600k_mbf.onnx")
    parser.add_argument("--faces-dir",         type=str,   default="./faces")
    parser.add_argument("--source",            type=str,   default="./assets/in_video.mp4")
    parser.add_argument("--similarity-thresh", type=float, default=0.4)
    parser.add_argument("--confidence-thresh", type=float, default=0.5)
    parser.add_argument("--max-frames",        type=int,   default=0, help="0 = entire video")
    return parser.parse_args()


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_targets(detector, recognizer, faces_dir):
    import os
    from pathlib import Path
    targets = []
    for filename in os.listdir(faces_dir):
        if Path(filename).suffix.lower() not in IMAGE_EXTS:
            continue
        name = Path(filename).stem
        image = cv2.imread(os.path.join(faces_dir, filename))
        if image is None:
            continue
        bboxes, kpss = detector.detect(image, max_num=1)
        if kpss is None or len(kpss) == 0:
            continue
        targets.append((recognizer(image, kpss[0]), name))
    return targets


def identify(embedding, targets, similarity_thresh):
    best_name, best_sim = "Unknown", 0.0
    for target_emb, name in targets:
        sim = compute_similarity(target_emb, embedding)
        if sim > best_sim and sim > similarity_thresh:
            best_sim = sim
            best_name = name
    return best_name, best_sim


def main():
    args = parse_args()

    print("=" * 55)
    print("VIDEO BENCHMARK")
    print("=" * 55)
    print(f"  Source  : {args.source}")
    print(f"  Det     : {args.det_weight}")
    print(f"  Rec     : {args.rec_weight}")
    print(f"  Thresh  : similarity={args.similarity_thresh}  confidence={args.confidence_thresh}")

    print("\nLoading models...")
    detector   = SCRFD(args.det_weight, input_size=(640, 640), conf_thres=args.confidence_thresh)
    recognizer = ArcFace(args.rec_weight)

    print("Building known face targets...")
    targets     = build_targets(detector, recognizer, args.faces_dir)
    known_names = {name for _, name in targets}
    print(f"  Known: {known_names}")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n  Video   : {width}x{height} @ {video_fps:.1f} FPS  ({total_frames} frames total)")
    limit = args.max_frames if args.max_frames > 0 else total_frames
    print(f"  Testing : {limit} frames")
    print("\nRunning...\n")

    # --- per-frame tracking ---
    fps_list         = []
    detection_counts = []          # faces detected per frame
    id_per_frame     = []          # list of sets of identities per frame
    prev_ids         = set()
    id_switch_count  = 0
    latency_list     = []

    frame_idx = 0
    while frame_idx < limit:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()

        bboxes, kpss = detector.detect(frame, max_num=0)
        current_ids  = set()

        if kpss is not None and len(kpss) > 0:
            for kps in kpss:
                emb        = recognizer(frame, kps)
                name, sim  = identify(emb, targets, args.similarity_thresh)
                current_ids.add(name)

        latency = (time.time() - t0) * 1000      # ms
        fps     = 1000.0 / latency if latency > 0 else 0

        fps_list.append(fps)
        latency_list.append(latency)
        detection_counts.append(len(kpss) if kpss is not None else 0)
        id_per_frame.append(current_ids)

        # ID switch: a known identity present last frame disappeared this frame
        for name in known_names:
            if name in prev_ids and name not in current_ids and len(current_ids) > 0:
                id_switch_count += 1
        prev_ids = current_ids

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  Processed {frame_idx}/{limit} frames  |  Current FPS: {fps:.1f}")

    cap.release()

    # --- compute stats ---
    fps_arr     = np.array(fps_list)
    lat_arr     = np.array(latency_list)
    det_arr     = np.array(detection_counts)

    frames_with_face    = int(np.sum(det_arr > 0))
    detection_rate      = frames_with_face / len(det_arr) * 100 if det_arr.size > 0 else 0

    # frames where GPU was stressed (multiple faces)
    multi_face_frames   = int(np.sum(det_arr > 1))

    print("\n" + "=" * 55)
    print("RESULTS")
    print("=" * 55)

    print(f"\n  Frames processed  : {frame_idx}")

    print(f"\n  FPS")
    print(f"    Average         : {fps_arr.mean():.1f}")
    print(f"    Min             : {fps_arr.min():.1f}")
    print(f"    Max             : {fps_arr.max():.1f}")
    print(f"    Std dev         : {fps_arr.std():.1f}")
    print(f"    Real-time (≥24) : {'YES ✓' if fps_arr.mean() >= 24 else 'NO ✗'}")

    print(f"\n  Latency (per frame)")
    print(f"    Average         : {lat_arr.mean():.1f} ms")
    print(f"    Min             : {lat_arr.min():.1f} ms")
    print(f"    Max             : {lat_arr.max():.1f} ms")
    print(f"    P95             : {np.percentile(lat_arr, 95):.1f} ms")

    print(f"\n  Face Detection")
    print(f"    Frames w/ face  : {frames_with_face}/{frame_idx}  ({detection_rate:.1f}%)")
    print(f"    Multi-face frames: {multi_face_frames}")
    print(f"    Avg faces/frame : {det_arr.mean():.2f}")

    print(f"\n  Identity Stability")
    print(f"    ID switches     : {id_switch_count}")
    stability = max(0, 100 - (id_switch_count / max(frames_with_face, 1) * 100))
    print(f"    Stability score : {stability:.1f}%")

    print(f"\n  Video source FPS  : {video_fps:.1f}")
    print(f"  Inference FPS     : {fps_arr.mean():.1f}")
    headroom = fps_arr.mean() - video_fps
    print(f"  Headroom          : {headroom:+.1f} FPS {'(can handle faster cameras)' if headroom > 0 else '(dropping frames)'}")

    print("=" * 55)


if __name__ == "__main__":
    main()
