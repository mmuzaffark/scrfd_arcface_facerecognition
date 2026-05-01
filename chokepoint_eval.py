import cv2
import time
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

from models import ArcFace
from utils.helpers import compute_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="ChokePoint Dataset Evaluation")
    parser.add_argument("--rec-weight",        type=str,   default="./weights/w600k_mbf.onnx")
    parser.add_argument("--chokepoint-dir",    type=str,   default="./chokepoint")
    parser.add_argument("--gallery-ratio",     type=float, default=0.5,  help="Fraction of frames used as gallery")
    parser.add_argument("--similarity-thresh", type=float, default=0.21, help="Best threshold from LFW")
    return parser.parse_args()


def load_pgm_as_bgr(image_path):
    """Load a PGM grayscale image and convert to BGR for ArcFace."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def get_embedding(recognizer, image_bgr):
    """Get ArcFace embedding directly from pre-cropped face image."""
    resized = cv2.resize(image_bgr, recognizer.input_size)
    return recognizer.get_feat([resized]).flatten()


def parse_groundtruth_xml(xml_path):
    """Parse XML to get frame -> person_id mapping."""
    tree   = ET.parse(xml_path)
    root   = tree.getroot()
    frames = {}
    for frame in root.findall("frame"):
        frame_num = frame.get("number")
        person    = frame.find("person")
        if person is not None:
            frames[frame_num] = person.get("id")
    return frames


def load_session(session_dir, gallery_ratio):
    """
    Load gallery and probe sets from a session directory.
    Each person subfolder contains frame images.
    Gallery = first gallery_ratio of frames per person.
    Probe   = remaining frames.
    """
    gallery = []  # (embedding_placeholder, person_id, path)
    probe   = []

    session_path = Path(session_dir)
    for person_dir in sorted(session_path.iterdir()):
        if not person_dir.is_dir():
            continue
        person_id = person_dir.name
        frames    = sorted(person_dir.glob("*.pgm"))
        if len(frames) == 0:
            continue

        split       = max(1, int(len(frames) * gallery_ratio))
        gallery_frames = frames[:split]
        probe_frames   = frames[split:]

        for f in gallery_frames:
            gallery.append((person_id, f))
        for f in probe_frames:
            probe.append((person_id, f))

    return gallery, probe


def build_gallery_embeddings(recognizer, gallery):
    """Extract embeddings for all gallery images, average per person."""
    person_embeddings = defaultdict(list)
    for person_id, path in gallery:
        img = load_pgm_as_bgr(path)
        if img is None:
            continue
        emb = get_embedding(recognizer, img)
        person_embeddings[person_id].append(emb)

    # Average embeddings per person for better representation
    averaged = {}
    for person_id, embs in person_embeddings.items():
        averaged[person_id] = np.mean(embs, axis=0)
    return averaged


def identify(embedding, gallery_embeddings, similarity_thresh):
    best_name, best_sim = "Unknown", 0.0
    for person_id, ref_emb in gallery_embeddings.items():
        sim = compute_similarity(ref_emb, embedding)
        if sim > best_sim and sim > similarity_thresh:
            best_sim  = sim
            best_name = person_id
    return best_name, best_sim


def evaluate_session(recognizer, session_dir, gallery_ratio, similarity_thresh):
    gallery, probe = load_session(session_dir, gallery_ratio)
    if not gallery or not probe:
        return None

    gallery_embeddings = build_gallery_embeddings(recognizer, gallery)

    results  = []
    timings  = []
    for person_id, path in probe:
        img = load_pgm_as_bgr(path)
        if img is None:
            continue

        t0  = time.time()
        emb = get_embedding(recognizer, img)
        pred, sim = identify(emb, gallery_embeddings, similarity_thresh)
        timings.append(time.time() - t0)

        results.append((person_id, pred, sim))

    return results, timings, gallery_embeddings


def compute_metrics(results, known_ids):
    labels = [gt for gt, _, _ in results]
    preds  = [pred for _, pred, _ in results]

    correct  = sum(1 for gt, pred, _ in results if gt == pred)
    accuracy = correct / len(results) if results else 0

    per_class = {}
    for cls in known_ids:
        tp = sum(1 for gt, pred, _ in results if gt == cls and pred == cls)
        fp = sum(1 for gt, pred, _ in results if gt != cls and pred == cls)
        fn = sum(1 for gt, pred, _ in results if gt == cls and pred != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class[cls] = {"TP": tp, "FP": fp, "FN": fn,
                          "Precision": precision, "Recall": recall, "F1": f1}
    return per_class, accuracy, correct


def main():
    args = parse_args()

    print("=" * 55)
    print("CHOKEPOINT DATASET EVALUATION")
    print("=" * 55)
    print(f"  Rec model        : {args.rec_weight}")
    print(f"  Chokepoint dir   : {args.chokepoint_dir}")
    print(f"  Gallery ratio    : {args.gallery_ratio}")
    print(f"  Similarity thresh: {args.similarity_thresh}")

    print("\nLoading ArcFace model...")
    recognizer = ArcFace(args.rec_weight)

    # Find all session directories
    chokepoint_path = Path(args.chokepoint_dir)
    session_dirs    = sorted([d for d in chokepoint_path.iterdir()
                              if d.is_dir() and d.name != "groundtruth"])

    print(f"\nFound {len(session_dirs)} sessions: {[d.name for d in session_dirs]}")

    all_results  = []
    all_timings  = []
    all_known_ids = set()

    for session_dir in session_dirs:
        print(f"\n  Processing {session_dir.name}...")
        output = evaluate_session(recognizer, session_dir,
                                  args.gallery_ratio, args.similarity_thresh)
        if output is None:
            print(f"    Skipped (empty)")
            continue

        results, timings, gallery_embeddings = output
        all_results.extend(results)
        all_timings.extend(timings)
        all_known_ids.update(gallery_embeddings.keys())

        correct  = sum(1 for gt, pred, _ in results if gt == pred)
        acc      = correct / len(results) if results else 0
        avg_fps  = 1.0 / np.mean(timings) if timings else 0
        print(f"    Frames : {len(results)}  |  Accuracy: {acc:.2%}  |  FPS: {avg_fps:.1f}")

    print("\n" + "=" * 55)
    print("OVERALL RESULTS")
    print("=" * 55)

    per_class, accuracy, correct = compute_metrics(all_results, all_known_ids)

    print(f"\n  Total probe frames : {len(all_results)}")
    print(f"  Known identities   : {len(all_known_ids)}")
    print(f"\n  Overall Accuracy   : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Correct / Total    : {correct}/{len(all_results)}")

    print(f"\n  Per-person metrics:")
    print(f"  {'ID':<8} {'Precision':>10} {'Recall':>10} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    print(f"  {'-'*55}")
    for cls in sorted(per_class.keys()):
        m = per_class[cls]
        print(f"  {cls:<8} {m['Precision']:>10.2f} {m['Recall']:>10.2f} "
              f"{m['F1']:>8.2f} {m['TP']:>5} {m['FP']:>5} {m['FN']:>5}")

    avg_precision = np.mean([m["Precision"] for m in per_class.values()])
    avg_recall    = np.mean([m["Recall"]    for m in per_class.values()])
    avg_f1        = np.mean([m["F1"]        for m in per_class.values()])
    print(f"\n  Macro Precision  : {avg_precision:.4f}")
    print(f"  Macro Recall     : {avg_recall:.4f}")
    print(f"  Macro F1         : {avg_f1:.4f}")

    avg_time = np.mean(all_timings) * 1000
    fps      = 1000.0 / avg_time if avg_time > 0 else 0
    print(f"\n  Speed:")
    print(f"    Avg time/frame : {avg_time:.1f} ms")
    print(f"    FPS            : {fps:.1f}")

    print(f"\n  Comparison:")
    print(f"    LFW Accuracy   : 99.56%")
    print(f"    Chokepoint Acc : {accuracy*100:.2f}%")
    print("=" * 55)


if __name__ == "__main__":
    main()
