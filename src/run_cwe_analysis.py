"""Run the per-CWE analysis pipeline.

1. Generate PU splits for seeds 789, 999 at 20% labeling
2. Train all 4 methods at 20% labeling for seeds 42, 123, 456, 789, 999
3. Evaluate each with --output-json
4. Compute pooled category recall directly from test predictions
5. Write experiments/logs/cwe_results.csv

Usage:
    python src/run_cwe_analysis.py
"""

import csv
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from model import VulnMLP
from sklearn.metrics import f1_score as sk_f1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 999]
FRAC = 0.20
TRUE_PRIOR = 0.0277
CHECKPOINT_DIR = Path("checkpoints")
JSON_DIR = Path("experiments/cwe_json")
CWE_CSV = Path("experiments/logs/cwe_results.csv")

# CWE pooling taxonomy
CWE_CATEGORIES = {
    "Memory safety": [
        "CWE-119", "CWE-125", "CWE-787", "CWE-415", "CWE-416", "CWE-190",
        "CWE-120", "CWE-122", "CWE-401", "CWE-772", "CWE-824", "CWE-823",
        "CWE-763", "CWE-908",
    ],
    "Logic / semantic": [
        "CWE-20", "CWE-476", "CWE-703", "CWE-617", "CWE-754", "CWE-252",
        "CWE-835", "CWE-834",
    ],
    "Concurrency": ["CWE-362"],
}


def run_cmd(cmd, timeout=600):
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        logger.error("FAILED: %s\n%s", " ".join(cmd), result.stderr[-300:])
        return False
    return True


def get_method_commands(seed):
    """Return dict of method_name -> (train_cmd, checkpoint_path)."""
    return {
        "supervised": (
            [sys.executable, "src/train_supervised.py", "--seed", str(seed)],
            CHECKPOINT_DIR / "supervised_seed{}.pt".format(seed),
        ),
        "naive_pu": (
            [sys.executable, "src/train_naive_pu.py",
             "--labeled-frac", str(FRAC), "--split-seed", str(seed), "--seed", str(seed)],
            CHECKPOINT_DIR / "naive_pu_frac{:.2f}_seed{}.pt".format(FRAC, seed),
        ),
        "nnpu": (
            [sys.executable, "src/train_nnpu.py",
             "--labeled-frac", str(FRAC), "--split-seed", str(seed), "--seed", str(seed),
             "--prior", str(TRUE_PRIOR)],
            CHECKPOINT_DIR / "nnpu_frac{:.2f}_prior{:.4f}_seed{}.pt".format(FRAC, TRUE_PRIOR, seed),
        ),
        "self_training": (
            [sys.executable, "src/train_self_training.py",
             "--labeled-frac", str(FRAC), "--split-seed", str(seed), "--seed", str(seed)],
            CHECKPOINT_DIR / "self_training_frac{:.2f}_seed{}.pt".format(FRAC, seed),
        ),
    }


def main():
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate PU splits for new seeds
    logger.info("=== Step 1: Generate PU splits for new seeds ===")
    for seed in [789, 999]:
        split_dir = Path("data/processed/frac{:.2f}_seed{}".format(FRAC, seed))
        if split_dir.exists():
            logger.info("Split seed=%d already exists", seed)
            continue
        run_cmd([sys.executable, "data/prepare_pu_splits.py",
                 "--labeled-frac", str(FRAC), "--seed", str(seed)])

    # Step 2: Train and evaluate all methods at all seeds
    logger.info("=== Step 2: Train and evaluate (5 seeds x 4 methods = 20 runs) ===")
    for seed in SEEDS:
        methods = get_method_commands(seed)
        for method_name, (train_cmd, ckpt) in methods.items():
            json_path = JSON_DIR / "{}_{}.json".format(method_name, seed)
            if json_path.exists():
                logger.info("Already have %s", json_path.name)
                continue

            if not ckpt.exists():
                logger.info("Training %s seed=%d", method_name, seed)
                if not run_cmd(train_cmd):
                    continue

            logger.info("Evaluating %s seed=%d -> %s", method_name, seed, json_path.name)
            run_cmd([
                sys.executable, "src/evaluate.py",
                "--checkpoint", str(ckpt),
                "--tune-threshold",
                "--output-json", str(json_path),
            ])

    # Step 3: Compute pooled category recall directly from test predictions
    logger.info("=== Step 3: Compute pooled category recalls ===")

    test = np.load("data/processed/embeddings/test.npz", allow_pickle=False)
    X_test = torch.tensor(test["embeddings"], dtype=torch.float32)
    y_test = test["labels"]
    test_cwes = test["cwe_types"]
    vuln_mask = y_test == 1
    vuln_cwes = test_cwes[vuln_mask]

    val_data = np.load("data/processed/embeddings/valid.npz", allow_pickle=False)
    X_val = torch.tensor(val_data["embeddings"], dtype=torch.float32)
    y_val = val_data["labels"]

    cat_rows = []
    for seed in SEEDS:
        methods = get_method_commands(seed)
        for method_name, (_, ckpt) in methods.items():
            if not ckpt.exists():
                logger.warning("Missing checkpoint %s", ckpt)
                continue

            model = VulnMLP(input_dim=768, hidden_dim=256, dropout=0.3)
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            model.eval()

            with torch.no_grad():
                vp = model.predict_proba(X_val).numpy()
                tp = model.predict_proba(X_test).numpy()

            # Tune threshold on val
            best_thr, best_f1 = 0.5, 0.0
            for thr in np.arange(0.05, 0.95, 0.01):
                f = sk_f1(y_val, (vp >= thr).astype(int))
                if f > best_f1:
                    best_f1 = f
                    best_thr = thr

            vuln_preds = (tp >= best_thr).astype(int)[vuln_mask]

            for cat_name, cat_cwes in CWE_CATEGORIES.items():
                cat_mask = np.isin(vuln_cwes, cat_cwes)
                n = int(cat_mask.sum())
                if n > 0:
                    recall = float(vuln_preds[cat_mask].mean())
                    cat_rows.append({
                        "method": method_name,
                        "seed": seed,
                        "category": cat_name,
                        "n_test": n,
                        "recall": round(recall, 4),
                    })

    # Write CSV
    CWE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CWE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "seed", "category", "n_test", "recall"])
        writer.writeheader()
        for row in cat_rows:
            writer.writerow(row)
    logger.info("Wrote %d rows to %s", len(cat_rows), CWE_CSV)

    # Print summary
    logger.info("\n=== Pooled Category Recall (mean +/- std across 5 seeds) ===\n")
    for cat in ["Memory safety", "Logic / semantic", "Concurrency"]:
        logger.info("%s:", cat)
        for method in ["supervised", "naive_pu", "nnpu", "self_training"]:
            vals = [r["recall"] for r in cat_rows
                    if r["category"] == cat and r["method"] == method]
            if vals:
                logger.info("  %-15s  %.3f +/- %.3f  (n_seeds=%d)",
                            method, np.mean(vals), np.std(vals), len(vals))
        logger.info("")


if __name__ == "__main__":
    main()
