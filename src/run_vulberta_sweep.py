"""Run labeling sweep on VulBERTa embeddings.

3 PU methods x 7 fractions x 3 seeds = 63 runs.
Plus 3 supervised runs for the ceiling.

Results appended to experiments/logs/vulberta_results.csv.

Usage:
    EMBEDDING_DIR=data/processed/embeddings_vulberta python src/run_vulberta_sweep.py
"""

import csv
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Force VulBERTa embeddings
os.environ["EMBEDDING_DIR"] = "data/processed/embeddings_vulberta"

RESULTS_FILE = Path("experiments/logs/vulberta_results.csv")
CHECKPOINT_DIR = Path("checkpoints")
FIELDS = [
    "experiment", "method", "labeled_frac", "split_seed", "model_seed",
    "prior", "threshold", "f1", "auroc", "auprc", "vds", "pairwise_acc",
]

SEEDS = [42, 123, 456]
FRACS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80]
TRUE_PRIOR = 0.0277


def run_cmd(cmd, timeout=600):
    env = dict(os.environ)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    if result.returncode != 0:
        logger.error("FAILED: %s\n%s", " ".join(cmd), result.stderr[-300:])
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
                  labeled_frac=0.20, split_seed=42, model_seed=42, prior=None):
    logger.info(">>> %s | %s | frac=%.2f | seed=%d", experiment, method, labeled_frac, split_seed)

    output = run_cmd(train_cmd)
    if not output:
        return

    ev_out = run_cmd([sys.executable, "src/evaluate.py",
                      "--checkpoint", str(checkpoint), "--tune-threshold"])
    if not ev_out:
        return

    metrics = parse_metrics(ev_out)
    if not metrics:
        logger.error("Could not parse metrics")
        return

    row = {
        "experiment": experiment,
        "method": method,
        "labeled_frac": labeled_frac,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "prior": prior if prior else "",
    }
    row.update(metrics)

    file_exists = RESULTS_FILE.exists() and RESULTS_FILE.stat().st_size > 0
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info("    F1=%.4f AUROC=%.4f", metrics.get("f1", 0), metrics.get("auroc", 0))


def main():
    if RESULTS_FILE.exists():
        RESULTS_FILE.unlink()
        logger.info("Cleared previous VulBERTa results")

    # Supervised ceiling (3 seeds)
    logger.info("=" * 60)
    logger.info("SUPERVISED CEILING (3 runs)")
    logger.info("=" * 60)
    for seed in SEEDS:
        ckpt = CHECKPOINT_DIR / "supervised_seed{}.pt".format(seed)
        train_and_log("vb_baseline", "supervised",
                      [sys.executable, "src/train_supervised.py", "--seed", str(seed)],
                      ckpt, labeled_frac=1.0, split_seed=seed, model_seed=seed)

    # Labeling sweep: 3 PU methods x 7 fracs x 3 seeds
    logger.info("=" * 60)
    logger.info("LABELING SWEEP (63 runs)")
    logger.info("=" * 60)
    for frac in FRACS:
        for seed in SEEDS:
            # Naive PU
            ckpt = CHECKPOINT_DIR / "naive_pu_frac{:.2f}_seed{}.pt".format(frac, seed)
            train_and_log("vb_sweep", "naive_pu",
                          [sys.executable, "src/train_naive_pu.py",
                           "--labeled-frac", str(frac), "--split-seed", str(seed), "--seed", str(seed)],
                          ckpt, labeled_frac=frac, split_seed=seed, model_seed=seed)

            # nnPU
            ckpt = CHECKPOINT_DIR / "nnpu_frac{:.2f}_prior{:.4f}_seed{}.pt".format(frac, TRUE_PRIOR, seed)
            train_and_log("vb_sweep", "nnpu",
                          [sys.executable, "src/train_nnpu.py",
                           "--labeled-frac", str(frac), "--split-seed", str(seed), "--seed", str(seed),
                           "--prior", str(TRUE_PRIOR)],
                          ckpt, labeled_frac=frac, split_seed=seed, model_seed=seed, prior=TRUE_PRIOR)

            # Self-training
            ckpt = CHECKPOINT_DIR / "self_training_frac{:.2f}_seed{}.pt".format(frac, seed)
            train_and_log("vb_sweep", "self_training",
                          [sys.executable, "src/train_self_training.py",
                           "--labeled-frac", str(frac), "--split-seed", str(seed), "--seed", str(seed)],
                          ckpt, labeled_frac=frac, split_seed=seed, model_seed=seed)

    total = 0
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            total = sum(1 for _ in f) - 1
    logger.info("=" * 60)
    logger.info("COMPLETE: %d results in %s", total, RESULTS_FILE)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
