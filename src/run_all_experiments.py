"""Run all Phase 3 experiments and log results to CSV.

Executes:
  Experiment 1: Baseline comparison (4 methods x 3 seeds at 20%)
  Experiment 2: Labeling budget sweep (3 methods x 7 fracs x 3 seeds)
  Experiment 3: Class prior sensitivity (nnPU x 6 priors x 3 seeds)
  Variance decomposition: 3 splits x 3 model seeds for nnPU + self-training

All results are appended to experiments/logs/all_results.csv.

Usage:
    python src/run_all_experiments.py
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
PRIORS = [0.0139, 0.0208, 0.0277, 0.0346, 0.0416, 0.0554]  # 0.5x to 2x
TRUE_PRIOR = 0.0277


def run_cmd(cmd, timeout=600):
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        logger.error("FAILED: %s\n%s", " ".join(cmd), result.stderr[-500:])
        return ""
    return result.stdout + result.stderr


def parse_metrics(output):
    """Extract metrics from the combined stdout+stderr of evaluate.py."""
    metrics = {}
    for line in output.split("\n"):
        if "Tuned threshold" in line:
            parts = line.split("threshold on val set: ")
            if len(parts) > 1:
                thr = parts[1].split(" ")[0]
                metrics["threshold"] = float(thr)
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


def train_and_log(
    experiment, method, train_cmd, checkpoint,
    labeled_frac=0.20, split_seed=42, model_seed=42,
    prior=None, neg_selection_pct=None,
):
    """Train, evaluate with tuned threshold, and log to CSV."""
    logger.info(">>> %s | %s | frac=%.2f | split=%d | seed=%d",
                experiment, method, labeled_frac, split_seed, model_seed)

    # Train
    output = run_cmd(train_cmd)
    if not output:
        return

    # Evaluate with tuned threshold
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

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RESULTS_FILE.exists() and RESULTS_FILE.stat().st_size > 0
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info("    F1=%.4f AUROC=%.4f thr=%.2f",
                metrics.get("f1", 0), metrics.get("auroc", 0), metrics.get("threshold", 0))


def run_experiment_1():
    """Baseline comparison: 4 methods x 3 seeds at 20% labeling."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Baseline Comparison (12 runs)")
    logger.info("=" * 60)

    frac = 0.20
    for seed in SEEDS:
        # Supervised
        ckpt = CHECKPOINT_DIR / "supervised_seed{}.pt".format(seed)
        train_and_log(
            "exp1_baseline", "supervised",
            [sys.executable, "src/train_supervised.py", "--seed", str(seed)],
            ckpt, labeled_frac=1.0, split_seed=seed, model_seed=seed,
        )

        # Naive PU
        ckpt = CHECKPOINT_DIR / "naive_pu_frac{:.2f}_seed{}.pt".format(frac, seed)
        train_and_log(
            "exp1_baseline", "naive_pu",
            [sys.executable, "src/train_naive_pu.py",
             "--labeled-frac", str(frac), "--split-seed", str(seed), "--seed", str(seed)],
            ckpt, labeled_frac=frac, split_seed=seed, model_seed=seed,
        )

        # nnPU
        ckpt = CHECKPOINT_DIR / "nnpu_frac{:.2f}_prior{:.4f}_seed{}.pt".format(frac, TRUE_PRIOR, seed)
        train_and_log(
            "exp1_baseline", "nnpu",
            [sys.executable, "src/train_nnpu.py",
             "--labeled-frac", str(frac), "--split-seed", str(seed), "--seed", str(seed),
             "--prior", str(TRUE_PRIOR)],
            ckpt, labeled_frac=frac, split_seed=seed, model_seed=seed, prior=TRUE_PRIOR,
        )

        # Self-training
        ckpt = CHECKPOINT_DIR / "self_training_frac{:.2f}_seed{}.pt".format(frac, seed)
        train_and_log(
            "exp1_baseline", "self_training",
            [sys.executable, "src/train_self_training.py",
             "--labeled-frac", str(frac), "--split-seed", str(seed), "--seed", str(seed)],
            ckpt, labeled_frac=frac, split_seed=seed, model_seed=seed,
        )


def run_experiment_2():
    """Labeling budget sweep: 3 PU methods x 7 fracs x 3 seeds."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Labeling Budget Sweep (63 runs)")
    logger.info("=" * 60)

    for frac in FRACS:
        for seed in SEEDS:
            # Naive PU
            ckpt = CHECKPOINT_DIR / "naive_pu_frac{:.2f}_seed{}.pt".format(frac, seed)
            train_and_log(
                "exp2_labeling", "naive_pu",
                [sys.executable, "src/train_naive_pu.py",
                 "--labeled-frac", str(frac), "--split-seed", str(seed), "--seed", str(seed)],
                ckpt, labeled_frac=frac, split_seed=seed, model_seed=seed,
            )

            # nnPU
            ckpt = CHECKPOINT_DIR / "nnpu_frac{:.2f}_prior{:.4f}_seed{}.pt".format(frac, TRUE_PRIOR, seed)
            train_and_log(
                "exp2_labeling", "nnpu",
                [sys.executable, "src/train_nnpu.py",
                 "--labeled-frac", str(frac), "--split-seed", str(seed), "--seed", str(seed),
                 "--prior", str(TRUE_PRIOR)],
                ckpt, labeled_frac=frac, split_seed=seed, model_seed=seed, prior=TRUE_PRIOR,
            )

            # Self-training
            ckpt = CHECKPOINT_DIR / "self_training_frac{:.2f}_seed{}.pt".format(frac, seed)
            train_and_log(
                "exp2_labeling", "self_training",
                [sys.executable, "src/train_self_training.py",
                 "--labeled-frac", str(frac), "--split-seed", str(seed), "--seed", str(seed)],
                ckpt, labeled_frac=frac, split_seed=seed, model_seed=seed,
            )


def run_experiment_3():
    """Class prior sensitivity: nnPU x 6 priors x 3 seeds at 20%."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: Class Prior Sensitivity (18 runs)")
    logger.info("=" * 60)

    frac = 0.20
    for prior in PRIORS:
        for seed in SEEDS:
            ckpt = CHECKPOINT_DIR / "nnpu_frac{:.2f}_prior{:.4f}_seed{}.pt".format(frac, prior, seed)
            train_and_log(
                "exp3_prior", "nnpu",
                [sys.executable, "src/train_nnpu.py",
                 "--labeled-frac", str(frac), "--split-seed", str(seed), "--seed", str(seed),
                 "--prior", str(prior)],
                ckpt, labeled_frac=frac, split_seed=seed, model_seed=seed, prior=prior,
            )


def run_variance_decomposition():
    """3 split seeds x 3 model seeds at 20% for nnPU and self-training.
    The diagonal (split_seed == model_seed) is already in exp1."""
    logger.info("=" * 60)
    logger.info("VARIANCE DECOMPOSITION: off-diagonal runs (12 runs)")
    logger.info("=" * 60)

    frac = 0.20
    split_seeds = [42, 123, 456]
    model_seeds = [42, 123, 456]

    for split_seed in split_seeds:
        for model_seed in model_seeds:
            if split_seed == model_seed:
                continue  # Already covered in exp1

            # nnPU
            ckpt = CHECKPOINT_DIR / "nnpu_var_s{}_m{}.pt".format(split_seed, model_seed)
            train_and_log(
                "variance_decomp", "nnpu",
                [sys.executable, "src/train_nnpu.py",
                 "--labeled-frac", str(frac), "--split-seed", str(split_seed),
                 "--seed", str(model_seed), "--prior", str(TRUE_PRIOR)],
                ckpt, labeled_frac=frac, split_seed=split_seed,
                model_seed=model_seed, prior=TRUE_PRIOR,
            )

            # Self-training
            ckpt = CHECKPOINT_DIR / "st_var_s{}_m{}.pt".format(split_seed, model_seed)
            train_and_log(
                "variance_decomp", "self_training",
                [sys.executable, "src/train_self_training.py",
                 "--labeled-frac", str(frac), "--split-seed", str(split_seed),
                 "--seed", str(model_seed)],
                ckpt, labeled_frac=frac, split_seed=split_seed,
                model_seed=model_seed,
            )


if __name__ == "__main__":
    # Clear previous results
    if RESULTS_FILE.exists():
        RESULTS_FILE.unlink()
        logger.info("Cleared previous results")

    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    run_variance_decomposition()

    # Count results
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            n_rows = sum(1 for _ in f) - 1  # minus header
    else:
        n_rows = 0

    logger.info("=" * 60)
    logger.info("ALL EXPERIMENTS COMPLETE: %d results logged", n_rows)
    logger.info("Results: %s", RESULTS_FILE)
    logger.info("=" * 60)
