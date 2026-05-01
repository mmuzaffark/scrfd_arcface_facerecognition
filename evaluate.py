import os
import cv2
import time
import argparse
import numpy as np
from pathlib import Path

from models import SCRFD, ArcFace
from utils.helpers import compute_similarity


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate face recognition metrics + FPS")
    parser.add_argument("--det-weight", type=str, default="./weights/det_500m.onnx")
    parser.add_argument("--rec-weight", type=str, default="./weights/w600k_mbf.onnx")
    parser.add_argument("--faces-dir", type=str, default="./faces")
    parser.add_argument("--test-dir", type=str, default="./test_faces")
    parser.add_argument("--similarity-thresh", type=float, default=0.4)
    parser.add_argument("--confidence-thresh", type=float, default=0.5)
    parser.add_argument("--video", type=str, default="./assets/in_video.mp4")
    parser.add_argument("--fps-frames", type=int, default=100, help="Number of frames to benchmark FPS on")
    return parser.parse_args()


def build_targets(detector, recognizer, faces_dir):
    targets = []
    abs_dir = str(Path(faces_dir).resolve())
    print(f"  Loading from: {abs_dir}")
    all_files = os.listdir(abs_dir)
    print(f"  Files found: {all_files}")
    for filename in all_files:
        path = os.path.join(abs_dir, filename)
        suffix = Path(filename).suffix.lower()
        if suffix not in IMAGE_EXTS:
            print(f"  [SKIP] {filename} (extension '{suffix}' not in {IMAGE_EXTS})")
            continue
        name = Path(filename).stem
        image = cv2.imread(path)
        if image is None:
            print(f"  [WARN] cv2.imread returned None for {path}")
            continue
        print(f"  [OK] Read {filename} shape={image.shape}")
        bboxes, kpss = detector.detect(image, max_num=1)
        if kpss is None or len(kpss) == 0:
            print(f"  [WARN] No face detected in {filename}, skipping")
            continue
        print(f"  [OK] Face detected in {filename}")
        targets.append((recognizer(image, kpss[0]), name))
    return targets


def predict_image(image, detector, recognizer, targets, similarity_thresh):
    bboxes, kpss = detector.detect(image, max_num=1)
    if len(kpss) == 0:
        return "No Face", 0.0
    embedding = recognizer(image, kpss[0])
    best_name, best_sim = "Unknown", 0.0
    for target_emb, name in targets:
        sim = compute_similarity(target_emb, embedding)
        if sim > best_sim and sim > similarity_thresh:
            best_sim = sim
            best_name = name
    return best_name, best_sim


def run_evaluation(detector, recognizer, targets, test_dir, similarity_thresh):
    known_names = {name for _, name in targets}
    results = []

    for person_dir in sorted(Path(test_dir).iterdir()):
        if not person_dir.is_dir():
            continue
        ground_truth = person_dir.name
        # For people not in known faces, the correct prediction is "Unknown"
        effective_gt = ground_truth if ground_truth in known_names else "Unknown"

        print(f"\n  {ground_truth} (expect: {effective_gt}):")
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"    [WARN] Could not read {img_path.name}")
                continue
            pred, sim = predict_image(image, detector, recognizer, targets, similarity_thresh)
            correct = pred == effective_gt
            status = "✓" if correct else "✗"
            print(f"    [{status}] {img_path.name}: Predicted={pred}, Sim={sim:.3f}")
            results.append((ground_truth, effective_gt, pred, sim, str(img_path)))

    return results


def compute_metrics(results):
    all_effective_gts = sorted(set(egt for _, egt, _, _, _ in results))
    all_predictions   = sorted(set(pred for _, _, pred, _, _ in results))
    all_classes       = sorted(set(all_effective_gts + all_predictions))

    per_class = {}
    for cls in all_classes:
        tp = sum(1 for _, egt, pred, _, _ in results if egt == cls and pred == cls)
        fp = sum(1 for _, egt, pred, _, _ in results if egt != cls and pred == cls)
        fn = sum(1 for _, egt, pred, _, _ in results if egt == cls and pred != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls] = {"TP": tp, "FP": fp, "FN": fn, "Precision": precision, "Recall": recall, "F1": f1}

    correct  = sum(1 for _, egt, pred, _, _ in results if egt == pred)
    accuracy = correct / len(results) if results else 0.0
    return per_class, accuracy, correct


def benchmark_fps(detector, recognizer, targets, video_path, n_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Could not open {video_path}")
        return 0.0, 0

    times = []
    count = 0
    while count < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.time()
        bboxes, kpss = detector.detect(frame, max_num=0)
        for kps in kpss:
            emb = recognizer(frame, kps)
            for target_emb, _ in targets:
                compute_similarity(target_emb, emb)
        times.append(time.time() - t0)
        count += 1

    cap.release()
    avg_time = sum(times) / len(times) if times else 1.0
    return 1.0 / avg_time, count


def main():
    args = parse_args()

    print("=" * 50)
    print("Loading models...")
    detector   = SCRFD(args.det_weight, input_size=(640, 640), conf_thres=args.confidence_thresh)
    recognizer = ArcFace(args.rec_weight)

    print("Building known face targets from faces/...")
    targets     = build_targets(detector, recognizer, args.faces_dir)
    known_names = {name for _, name in targets}
    print(f"  Known: {known_names}")

    print("\n" + "=" * 50)
    print("Running evaluation on test_faces/...")
    results = run_evaluation(detector, recognizer, targets, args.test_dir, args.similarity_thresh)

    print("\n" + "=" * 50)
    print("METRICS")
    print("=" * 50)
    per_class, accuracy, correct = compute_metrics(results)

    for cls, m in sorted(per_class.items()):
        print(f"\n  {cls}:")
        print(f"    Precision : {m['Precision']:.2f}  (TP={m['TP']}, FP={m['FP']})")
        print(f"    Recall    : {m['Recall']:.2f}  (FN={m['FN']})")
        print(f"    F1 Score  : {m['F1']:.2f}")

    print(f"\n  Overall Accuracy : {accuracy:.2%}  ({correct}/{len(results)} correct)")

    print("\n" + "=" * 50)
    print(f"FPS BENCHMARK  ({args.fps_frames} frames from {args.video})")
    print("=" * 50)
    fps, frames = benchmark_fps(detector, recognizer, targets, args.video, args.fps_frames)
    print(f"  Average FPS : {fps:.1f}  (over {frames} frames)")
    print("=" * 50)


if __name__ == "__main__":
    main()
