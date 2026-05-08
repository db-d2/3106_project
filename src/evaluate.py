"""Evaluate a trained MLP checkpoint on PrimeVul test set.

Computes F1, VDS (FNR at FPR <= 0.5%), and pair-wise accuracy.

Usage:
    python src/evaluate.py --checkpoint checkpoints/supervised_seed42.pt
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from model import VulnMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import os; EMBEDDING_DIR = Path(os.environ.get("EMBEDDING_DIR", "data/processed/embeddings"))


def load_embeddings(split: str) -> dict:
    """Load embeddings, labels, and metadata for a split."""
    data = np.load(EMBEDDING_DIR / (split + ".npz"), allow_pickle=False)
    return {
        "embeddings": data["embeddings"],
        "labels": data["labels"],
        "cwe_types": data["cwe_types"],
        "idxs": data["idxs"],
    }


def compute_vds(
    probs: np.ndarray, labels: np.ndarray, target_fpr: float = 0.005
) -> float:
    """Compute Vulnerability Detection Score: FNR at FPR <= target_fpr.

    Matches the PrimeVul paper's official implementation: classify at the
    threshold that maximizes TPR while keeping FPR <= target_fpr, then
    compute FNR = FN / (FN + TP) from hard predictions.

    Lower is better (fewer missed vulnerabilities at the given FPR tolerance).
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)

    # roc_curve prepends a sentinel at index 0 (fpr=0, tpr=0, threshold=+inf).
    # Filter to real operating points where FPR is within budget.
    valid_indices = np.where(fpr <= target_fpr)[0]
    # Pick the point with the largest FPR still within budget (most permissive)
    idx = valid_indices[-1]
    chosen_threshold = thresholds[idx]

    # Match PrimeVul official: classify at threshold, compute FNR from counts
    classified = (probs >= chosen_threshold).astype(int)
    fn = int(((labels == 1) & (classified == 0)).sum())
    tp = int(((labels == 1) & (classified == 1)).sum())
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return fnr


def compute_pairwise_accuracy(model: VulnMLP, device: torch.device) -> float:
    """Compute pair-wise accuracy on matched vulnerable/patched function pairs.

    Each consecutive pair in test_paired is (vulnerable, patched).
    The model should assign higher probability to the vulnerable function.
    Returns -1 if paired data is not available.
    """
    paired_path = EMBEDDING_DIR / "test_paired.npz"
    if not paired_path.exists():
        logger.warning("test_paired.npz not found, skipping pairwise accuracy")
        return -1.0

    data = np.load(paired_path, allow_pickle=False)
    embeddings = data["embeddings"]
    labels = data["labels"]

    X = torch.tensor(embeddings, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(X).cpu().numpy()

    # Pairs are consecutive: (vuln, patched), (vuln, patched), ...
    correct = 0
    total = 0
    for i in range(0, len(probs) - 1, 2):
        if labels[i] == 1 and labels[i + 1] == 0:
            correct += int(probs[i] > probs[i + 1])
            total += 1
        elif labels[i] == 0 and labels[i + 1] == 1:
            correct += int(probs[i + 1] > probs[i])
            total += 1

    if total == 0:
        return -1.0
    return correct / total


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Find the threshold that maximizes F1 on the given data."""
    best_f1 = -1.0
    best_thr = 0.5
    for thr in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= thr).astype(int)
        score = f1_score(labels, preds)
        if score > best_f1:
            best_f1 = score
            best_thr = thr
    return best_thr, best_f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MLP checkpoint")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--output-json", type=Path, default=None,
        help="Write all metrics including per-CWE recall to a JSON file",
    )
    parser.add_argument(
        "--tune-threshold", action="store_true",
        help="Tune classification threshold on val set instead of using 0.5",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = VulnMLP(
        input_dim=768, hidden_dim=args.hidden_dim, dropout=args.dropout
    ).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    # Determine threshold
    threshold = 0.5
    if args.tune_threshold:
        val_data = load_embeddings("valid")
        X_val = torch.tensor(val_data["embeddings"], dtype=torch.float32).to(device)
        with torch.no_grad():
            val_probs = model.predict_proba(X_val).cpu().numpy()
        threshold, val_f1 = find_best_threshold(val_probs, val_data["labels"])
        logger.info("Tuned threshold on val set: %.2f (val F1: %.4f)", threshold, val_f1)
    else:
        logger.info("Using fixed threshold: %.2f", threshold)

    # Load test data
    test_data = load_embeddings("test")
    X_test = torch.tensor(test_data["embeddings"], dtype=torch.float32).to(device)
    y_test = test_data["labels"]

    # Get predictions
    with torch.no_grad():
        probs = model.predict_proba(X_test).cpu().numpy()
    preds = (probs >= threshold).astype(int)

    # F1
    f1 = f1_score(y_test, preds)
    logger.info("F1 (threshold=%.2f): %.4f", threshold, f1)

    # Classification report
    logger.info(
        "\n%s",
        classification_report(y_test, preds, target_names=["benign", "vulnerable"]),
    )

    # Threshold-free metrics
    auroc = roc_auc_score(y_test, probs)
    auprc = average_precision_score(y_test, probs)
    logger.info("AUROC: %.4f", auroc)
    logger.info("AUPRC: %.4f", auprc)

    # VDS
    vds = compute_vds(probs, y_test)
    logger.info("VDS (FNR at FPR <= 0.5%%): %.4f", vds)

    # Pair-wise accuracy
    pw_acc = compute_pairwise_accuracy(model, device)
    if pw_acc >= 0:
        logger.info("Pair-wise accuracy: %.4f", pw_acc)

    # Per-CWE breakdown (vulnerable samples only)
    cwe_types = test_data["cwe_types"]
    vuln_mask = y_test == 1
    per_cwe = {}
    if vuln_mask.sum() > 0:
        vuln_cwes = cwe_types[vuln_mask]
        vuln_preds = preds[vuln_mask]
        vuln_probs = probs[vuln_mask]
        unique_cwes, counts = np.unique(vuln_cwes, return_counts=True)

        logger.info("\nPer-CWE recall (CWEs with >= 5 test samples):")
        for cwe, count in sorted(zip(unique_cwes, counts), key=lambda x: -x[1]):
            if count >= 5:
                cwe_mask = vuln_cwes == cwe
                recall = float(vuln_preds[cwe_mask].mean())
                per_cwe[str(cwe)] = {"n": int(count), "recall": round(recall, 4)}
                logger.info("  %s (n=%d): recall=%.3f", cwe, count, recall)

    # Summary
    results = {
        "f1": round(float(f1), 4),
        "threshold": round(float(threshold), 2),
        "auroc": round(float(auroc), 4),
        "auprc": round(float(auprc), 4),
        "vds": round(float(vds), 4),
        "pairwise_acc": round(float(pw_acc), 4) if pw_acc >= 0 else None,
        "per_cwe": per_cwe,
    }
    logger.info("\nResults summary (aggregate): f1=%.4f auroc=%.4f auprc=%.4f",
                f1, auroc, auprc)

    # Write structured JSON if requested
    if args.output_json:
        import json
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as jf:
            json.dump(results, jf, indent=2)
        logger.info("Saved JSON to %s", args.output_json)


if __name__ == "__main__":
    main()
