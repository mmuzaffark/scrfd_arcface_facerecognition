import cv2
import time
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve

from models import SCRFD, ArcFace
from utils.helpers import compute_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="LFW Face Verification Benchmark")
    parser.add_argument("--det-weight",        type=str,   default="./weights/det_500m.onnx")
    parser.add_argument("--rec-weight",        type=str,   default="./weights/w600k_mbf.onnx")
    parser.add_argument("--lfw-dir",           type=str,   default="./lfw_dataset/lfw_funneled")
    parser.add_argument("--pairs-file",        type=str,   default="./lfw_dataset/pairs.txt")
    parser.add_argument("--confidence-thresh", type=float, default=0.5)
    parser.add_argument("--similarity-thresh", type=float, default=0.4)
    return parser.parse_args()


def get_embedding(image_path, detector, recognizer, conf_thresh):
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    bboxes, kpss = detector.detect(image, max_num=1)
    if kpss is None or len(kpss) == 0:
        return None
    return recognizer(image, kpss[0])


def load_pairs(pairs_file, lfw_dir):
    """
    Load LFW pairs.txt format:
    Line 1: num_folds  num_pairs_per_fold
    Matched pairs:   name  img_num1  img_num2
    Mismatched pairs: name1  img_num1  name2  img_num2
    """
    matched   = []
    mismatched = []

    with open(pairs_file, "r") as f:
        lines = f.read().strip().split("\n")

    first   = lines[0].split()
    n_folds = int(first[0])
    n_pairs = int(first[1])

    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) == 3:
            # matched pair — same person
            name, n1, n2 = parts
            img1 = Path(lfw_dir) / name / f"{name}_{int(n1):04d}.jpg"
            img2 = Path(lfw_dir) / name / f"{name}_{int(n2):04d}.jpg"
            matched.append((img1, img2, 1))
        elif len(parts) == 4:
            # mismatched pair — different people
            name1, n1, name2, n2 = parts
            img1 = Path(lfw_dir) / name1 / f"{name1}_{int(n1):04d}.jpg"
            img2 = Path(lfw_dir) / name2 / f"{name2}_{int(n2):04d}.jpg"
            mismatched.append((img1, img2, 0))

    return matched, mismatched


def main():
    args = parse_args()

    print("=" * 55)
    print("LFW FACE VERIFICATION BENCHMARK")
    print("=" * 55)
    print(f"  Det model : {args.det_weight}")
    print(f"  Rec model : {args.rec_weight}")
    print(f"  LFW dir   : {args.lfw_dir}")
    print(f"  Pairs file: {args.pairs_file}")

    print("\nLoading models...")
    detector   = SCRFD(args.det_weight, input_size=(640, 640), conf_thres=args.confidence_thresh)
    recognizer = ArcFace(args.rec_weight)

    print("Loading pairs...")
    matched, mismatched = load_pairs(args.pairs_file, args.lfw_dir)
    all_pairs = matched + mismatched
    print(f"  Matched pairs    : {len(matched)}")
    print(f"  Mismatched pairs : {len(mismatched)}")
    print(f"  Total pairs      : {len(all_pairs)}")

    print("\nExtracting embeddings and computing similarities...")
    similarities = []
    labels       = []
    skipped      = 0
    timings      = []

    for i, (img1_path, img2_path, label) in enumerate(all_pairs):
        t0 = time.time()

        emb1 = get_embedding(img1_path, detector, recognizer, args.confidence_thresh)
        emb2 = get_embedding(img2_path, detector, recognizer, args.confidence_thresh)

        timings.append(time.time() - t0)

        if emb1 is None or emb2 is None:
            skipped += 1
            continue

        sim = compute_similarity(emb1, emb2)
        similarities.append(sim)
        labels.append(label)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(all_pairs)} pairs  |  Skipped: {skipped}")

    similarities = np.array(similarities)
    labels       = np.array(labels)

    print(f"\n  Done. Skipped {skipped} pairs (no face detected)")

    # --- find best threshold ---
    best_acc   = 0
    best_thresh = 0
    for thresh in np.arange(0.1, 0.9, 0.01):
        preds   = (similarities >= thresh).astype(int)
        acc     = np.mean(preds == labels)
        if acc > best_acc:
            best_acc    = acc
            best_thresh = thresh

    # --- metrics at best threshold ---
    preds = (similarities >= best_thresh).astype(int)
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    tn = np.sum((preds == 0) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    precision  = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall     = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # --- metrics at user threshold ---
    preds_user = (similarities >= args.similarity_thresh).astype(int)
    acc_user   = np.mean(preds_user == labels)

    # --- AUC ---
    try:
        auc = roc_auc_score(labels, similarities)
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        # TAR @ FAR=0.001
        tar_at_far = 0.0
        for f, t in zip(fpr, tpr):
            if f <= 0.001:
                tar_at_far = t
    except Exception:
        auc        = 0.0
        tar_at_far = 0.0

    # --- FPS ---
    avg_pair_time = np.mean(timings)
    fps           = 1.0 / avg_pair_time if avg_pair_time > 0 else 0

    print("\n" + "=" * 55)
    print("RESULTS")
    print("=" * 55)

    print(f"\n  Pairs evaluated  : {len(similarities)}/{len(all_pairs)}")

    print(f"\n  At best threshold ({best_thresh:.2f}):")
    print(f"    Accuracy   : {best_acc:.4f}  ({best_acc*100:.2f}%)")
    print(f"    Precision  : {precision:.4f}")
    print(f"    Recall     : {recall:.4f}")
    print(f"    F1 Score   : {f1:.4f}")
    print(f"    TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    print(f"\n  At your threshold ({args.similarity_thresh:.2f}):")
    print(f"    Accuracy   : {acc_user:.4f}  ({acc_user*100:.2f}%)")

    print(f"\n  AUC            : {auc:.4f}")
    print(f"  TAR @ FAR=0.001: {tar_at_far:.4f}  ({tar_at_far*100:.2f}%)")

    print(f"\n  Similarity stats:")
    print(f"    Same person avg  : {similarities[labels==1].mean():.4f}")
    print(f"    Diff person avg  : {similarities[labels==0].mean():.4f}")

    print(f"\n  Speed:")
    print(f"    Avg time/pair  : {avg_pair_time*1000:.1f} ms")
    print(f"    Pairs/sec      : {fps:.1f}")

    print("=" * 55)


if __name__ == "__main__":
    main()
