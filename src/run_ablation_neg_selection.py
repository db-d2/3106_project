"""Ablation: self-training neg-selection-pct at {5%, 10%, 20%, 50%}.

Tests how sensitive self-training is to the fraction of U selected as
reliable negatives each iteration. Run at 20% labeling, 3 seeds.

Results appended to experiments/logs/all_results.csv.

Usage:
    python src/run_ablation_neg_selection.py
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
NEG_PCTS = [0.05, 0.10, 0.20, 0.50]
FRAC = 0.20


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


def main():
    logger.info("=" * 60)
    logger.info("ABLATION: Self-training neg-selection-pct (%d runs)",
                len(NEG_PCTS) * len(SEEDS))
    logger.info("=" * 60)

    for pct in NEG_PCTS:
        for seed in SEEDS:
            logger.info(">>> neg_pct=%.2f | seed=%d", pct, seed)

            ckpt = CHECKPOINT_DIR / "st_negpct{:.2f}_seed{}.pt".format(pct, seed)

            # Train
            train_cmd = [
                sys.executable, "src/train_self_training.py",
                "--labeled-frac", str(FRAC),
                "--split-seed", str(seed),
                "--seed", str(seed),
                "--neg-selection-pct", str(pct),
            ]
            output = run_cmd(train_cmd)
            if not output:
                continue

            # The self-training script saves to a fixed checkpoint name based on
            # frac and seed. We need to rename it to include neg_pct.
            default_ckpt = CHECKPOINT_DIR / "self_training_frac{:.2f}_seed{}.pt".format(FRAC, seed)
            if default_ckpt.exists():
                default_ckpt.rename(ckpt)

            # Evaluate
            ev_out = run_cmd([
                sys.executable, "src/evaluate.py",
                "--checkpoint", str(ckpt),
                "--tune-threshold",
            ])
            if not ev_out:
                continue

            metrics = parse_metrics(ev_out)
            if not metrics:
                logger.error("Could not parse metrics for %s", ckpt)
                continue

            row = {
                "experiment": "ablation_neg_pct",
                "method": "self_training",
                "labeled_frac": FRAC,
                "split_seed": seed,
                "model_seed": seed,
                "prior": "",
                "neg_selection_pct": pct,
            }
            row.update(metrics)

            RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            file_exists = RESULTS_FILE.exists() and RESULTS_FILE.stat().st_size > 0
            with open(RESULTS_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

            logger.info("    F1=%.4f AUROC=%.4f neg_pct=%.2f",
                        metrics.get("f1", 0), metrics.get("auroc", 0), pct)

    logger.info("=" * 60)
    logger.info("ABLATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
