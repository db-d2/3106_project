"""Prepare PU (Positive-Unlabeled) splits from PrimeVul dataset.

Takes the full PrimeVul training data and produces a PU split:
- P: a sampled fraction of vulnerable functions (labeled positive)
- U: everything else (labels stripped)

Dev and test sets are left untouched with full labels.

Usage:
    python data/prepare_pu_splits.py --labeled-frac 0.2 --seed 42
"""

import argparse
import json
import logging
import random
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "raw" / "PrimeVul_v0.1"
OUT_DIR = Path(__file__).parent / "processed"


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def create_pu_split(
    records: list[dict], labeled_frac: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split training records into P (labeled positives) and U (unlabeled).

    Args:
        records: Full training records with 'target' field.
        labeled_frac: Fraction of vulnerable functions to keep as labeled P.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (P records, U records). U records have target set to -1.
    """
    rng = random.Random(seed)

    vulnerable = [r for r in records if r["target"] == 1]
    benign = [r for r in records if r["target"] == 0]

    n_labeled = max(1, int(len(vulnerable) * labeled_frac))
    rng.shuffle(vulnerable)

    p_records = []
    for r in vulnerable[:n_labeled]:
        p_rec = dict(r)
        p_rec["true_target"] = 1  # consistent with U records
        p_records.append(p_rec)
    unlabeled_vuln = vulnerable[n_labeled:]

    # U = remaining vulnerable (label stripped) + all benign (label stripped)
    u_records = []
    for r in unlabeled_vuln:
        u_rec = dict(r)
        u_rec["target"] = -1  # -1 = unlabeled
        u_rec["true_target"] = 1  # keep ground truth for debugging only
        u_records.append(u_rec)
    for r in benign:
        u_rec = dict(r)
        u_rec["target"] = -1
        u_rec["true_target"] = 0
        u_records.append(u_rec)

    return p_records, u_records


def save_jsonl(records: list[dict], path: Path) -> None:
    """Save records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PU splits from PrimeVul")
    parser.add_argument(
        "--labeled-frac",
        type=float,
        default=0.20,
        help="Fraction of vulnerable functions to label as P (default: 0.20)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Path to PrimeVul v0.1 data directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory for processed splits",
    )
    args = parser.parse_args()

    # Load training data
    train_path = args.data_dir / "primevul_train.jsonl"
    logger.info(f"Loading training data from {train_path}")
    train_records = load_jsonl(train_path)

    total = len(train_records)
    n_vuln = sum(1 for r in train_records if r["target"] == 1)
    n_benign = total - n_vuln
    logger.info(f"Loaded {total} records: {n_vuln} vulnerable, {n_benign} benign")
    logger.info(f"Class prior: {n_vuln / total:.4f}")

    # Create PU split
    logger.info(
        f"Creating PU split: labeled_frac={args.labeled_frac}, seed={args.seed}"
    )
    p_records, u_records = create_pu_split(
        train_records, args.labeled_frac, args.seed
    )

    n_hidden_vuln = sum(1 for r in u_records if r.get("true_target") == 1)
    logger.info(f"P (labeled positive): {len(p_records)} functions")
    logger.info(f"U (unlabeled): {len(u_records)} functions ({n_hidden_vuln} hidden vulnerable)")

    # Save PU split
    split_name = f"frac{args.labeled_frac:.2f}_seed{args.seed}"
    split_dir = args.out_dir / split_name

    save_jsonl(p_records, split_dir / "P.jsonl")
    save_jsonl(u_records, split_dir / "U.jsonl")
    logger.info(f"Saved PU split to {split_dir}")

    # Save split metadata
    metadata = {
        "labeled_frac": args.labeled_frac,
        "seed": args.seed,
        "n_P": len(p_records),
        "n_U": len(u_records),
        "n_hidden_vuln_in_U": n_hidden_vuln,
        "class_prior": n_vuln / total,
        "source": str(train_path),
    }
    with open(split_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {split_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
