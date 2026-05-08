"""Rerun experiments that need fixing.

1. Supervised + naive PU with AUROC early stopping (was F1@0.5)
2. Variance decomposition (checkpoint naming was wrong)

Overwrites affected rows in all_results.csv by appending with
experiment names that include '_v2' suffix, then we can filter.

Usage:
    python src/run_reruns.py
"""

import csv
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_FILE = Path("experiments/logs/all_results.csv")
CHECKPOINT_DIR = Path("checkpoints")
FIELDS = [
    "experiment", "method", "labeled_frac", "split_seed", "model_seed",
    "prior", "neg_selection_pct", "threshold", "f1", "auroc", "auprc",
    "vds", "pairwise_acc",
]

SEEDS = [42, 123, 456]
FRACS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80]
TRUE_PRIOR = 0.0277


def run_cmd(cmd, timeout=600):
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        logger.error("FAILED: %s\n%s", " ".join(cmd), result.stderr[-500:])
        return ""
    return result.stdout + result.stderr


def parse_metrics(output):
    metrics = {}
    for line in output.split("\n"):
        if "Tuned threshold" in line:
            parts = line.split("threshold on val set: ")
            if len(parts) > 1:
                metrics["threshold"] = float(parts[1].split(" ")[0])
        if "F1 (threshold=" in line:
            metrics["f1"] = float(line.split(": ")[-1])
        if "AUROC:" in line:
            metrics["auroc"] = float(line.split(": ")[-1])
        if "AUPRC:" in line:
            metrics["auprc"] = float(line.split(": ")[-1])
        if "VDS" in line and "FNR" in line:
            metrics["vds"] = float(line.split(": ")[-1])
        if "Pair-wise accuracy:" in line:
            metrics["pairwise_acc"] = float(line.split(": ")[-1])
    return metrics


def train_and_log(experiment, method, train_cmd, checkpoint,
                  labeled_frac=0.20, split_seed=42, model_seed=42,
                  prior=None, neg_selection_pct=None):
    logger.info(">>> %s | %s | frac=%.2f | split=%d | seed=%d",
                experiment, method, labeled_frac, split_seed, model_seed)

    output = run_cmd(train_cmd)
    if not output:
        return

    ev_out = run_cmd([
        sys.executable, "src/evaluate.py",
        "--checkpoint", str(checkpoint),
        "--tune-threshold",
    ])
    if not ev_out:
        return

    metrics = parse_metrics(ev_out)
    if not metrics:
        logger.error("Could not parse metrics for %s", checkpoint)
        return

    row = {
        "experiment": experiment,
        "method": method,
        "labeled_frac": labeled_frac,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "prior": prior if prior else "",
        "neg_selection_pct": neg_selection_pct if neg_selection_pct else "",
    }
    row.update(metrics)

    file_exists = RESULTS_FILE.exists() and RESULTS_FILE.stat().st_size > 0
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info("    F1=%.4f AUROC=%.4f thr=%.2f",
                metrics.get("f1", 0), metrics.get("auroc", 0), metrics.get("threshold", 0))


def rerun_supervised_naive():
    """Rerun supervised and naive PU with AUROC early stopping."""
    logger.info("=" * 60)
    logger.info("RERUN: Supervised + Naive PU (AUROC early stopping)")
    logger.info("=" * 60)

    # Exp1: 3 seeds
    for seed in SEEDS:
        ckpt = CHECKPOINT_DIR / "supervised_seed{}.pt".format(seed)
        train_and_log(
            "exp1_baseline_v2", "supervised",
            [sys.executable, "src/train_supervised.py", "--seed", str(seed)],
            ckpt, labeled_frac=1.0, split_seed=seed, model_seed=seed,
        )

        ckpt = CHECKPOINT_DIR / "naive_pu_frac0.20_seed{}.pt".format(seed)
        train_and_log(
            "exp1_baseline_v2", "naive_pu",
            [sys.executable, "src/train_naive_pu.py",
             "--labeled-frac", "0.20", "--split-seed", str(seed), "--seed", str(seed)],
            ckpt, labeled_frac=0.20, split_seed=seed, model_seed=seed,
        )

    # Exp2: naive PU at all fractions x 3 seeds
    for frac in FRACS:
        for seed in SEEDS:
            ckpt = CHECKPOINT_DIR / "naive_pu_frac{:.2f}_seed{}.pt".format(frac, seed)
            train_and_log(
                "exp2_labeling_v2", "naive_pu",
                [sys.executable, "src/train_naive_pu.py",
                 "--labeled-frac", str(frac), "--split-seed", str(seed), "--seed", str(seed)],
                ckpt, labeled_frac=frac, split_seed=seed, model_seed=seed,
            )


def run_variance_decomp():
    """Variance decomposition with correct checkpoint handling."""
    logger.info("=" * 60)
    logger.info("VARIANCE DECOMPOSITION (fixed checkpoint naming)")
    logger.info("=" * 60)

    frac = 0.20
    split_seeds = [42, 123, 456]
    model_seeds = [42, 123, 456]

    for split_seed in split_seeds:
        for model_seed in model_seeds:
            # nnPU: trains and saves to the default nnpu checkpoint name
            # which includes the model seed, so different model seeds don't collide
            ckpt = CHECKPOINT_DIR / "nnpu_frac{:.2f}_prior{:.4f}_seed{}.pt".format(
                frac, TRUE_PRIOR, model_seed)
            train_and_log(
                "variance_decomp", "nnpu",
                [sys.executable, "src/train_nnpu.py",
                 "--labeled-frac", str(frac), "--split-seed", str(split_seed),
                 "--seed", str(model_seed), "--prior", str(TRUE_PRIOR)],
                ckpt, labeled_frac=frac, split_seed=split_seed,
                model_seed=model_seed, prior=TRUE_PRIOR,
            )

            # Self-training
            ckpt = CHECKPOINT_DIR / "self_training_frac{:.2f}_seed{}.pt".format(
                frac, model_seed)
            train_and_log(
                "variance_decomp", "self_training",
                [sys.executable, "src/train_self_training.py",
                 "--labeled-frac", str(frac), "--split-seed", str(split_seed),
                 "--seed", str(model_seed)],
                ckpt, labeled_frac=frac, split_seed=split_seed,
                model_seed=model_seed,
            )


if __name__ == "__main__":
    rerun_supervised_naive()
    run_variance_decomp()

    total = 0
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            total = sum(1 for _ in f) - 1

    logger.info("=" * 60)
    logger.info("RERUNS COMPLETE. Total rows in CSV: %d", total)
    logger.info("=" * 60)
