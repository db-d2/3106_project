"""Train self-training (pseudo-labeling) model on CodeBERT embeddings.

Pseudo-labeling family. Prior-free.

Algorithm:
  1. Train initial MLP treating U as negative (same as naive PU).
  2. Use trained model to score all U samples.
  3. Select top-K most confident negatives from U (lowest predicted probability).
  4. Retrain MLP on P + selected reliable negatives.
  5. Repeat steps 2-4 for T iterations.

Usage:
    python src/train_self_training.py --labeled-frac 0.20 --seed 42
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from model import VulnMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import os; EMBEDDING_DIR = Path(os.environ.get("EMBEDDING_DIR", "data/processed/embeddings"))
SPLIT_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pu_data(labeled_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Load P and U embeddings separately."""
    split_name = "frac{:.2f}_seed{}".format(labeled_frac, seed)
    split_dir = SPLIT_DIR / split_name

    if not split_dir.exists():
        raise FileNotFoundError(
            "PU split not found: {}. Run: python data/prepare_pu_splits.py "
            "--labeled-frac {} --seed {}".format(split_dir, labeled_frac, seed)
        )

    train_data = np.load(EMBEDDING_DIR / "train.npz", allow_pickle=False)
    all_embeddings = train_data["embeddings"]
    all_idxs = train_data["idxs"]
    idx_to_pos = {int(idx): i for i, idx in enumerate(all_idxs)}

    p_indices = []
    with open(split_dir / "P.jsonl") as f:
        for line in f:
            p_indices.append(idx_to_pos[json.loads(line)["idx"]])

    u_indices = []
    with open(split_dir / "U.jsonl") as f:
        for line in f:
            u_indices.append(idx_to_pos[json.loads(line)["idx"]])

    X_p = all_embeddings[p_indices]
    X_u = all_embeddings[u_indices]
    return X_p, X_u


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    device: torch.device,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    lr: float = 1e-3,
    epochs: int = 20,
    batch_size: int = 512,
    patience: int = 5,
) -> VulnMLP:
    """Train an MLP and return the best model (by val F1)."""
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    X_v = torch.tensor(X_valid, dtype=torch.float32).to(device)

    model = VulnMLP(input_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_pos = max(int(y_train.sum()), 1)
    n_neg = len(y_train) - n_pos
    pw = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_f1 = -1.0
    best_state = None
    wait = 0

    for _ in range(1, epochs + 1):
        model.train()
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probs = model.predict_proba(X_v).cpu().numpy()
        val_f1 = f1_score(y_valid, (val_probs >= 0.5).astype(int))

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train self-training PU model")
    parser.add_argument("--labeled-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42, help="Model training seed")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="PU split seed (defaults to --seed)")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of self-training iterations")
    parser.add_argument("--neg-selection-pct", type=float, default=0.10,
                        help="Fraction of U to select as reliable negatives each iteration")
    parser.add_argument("--inner-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()
    if args.split_seed is None:
        args.split_seed = args.seed

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load data
    logger.info("Loading PU split: frac=%.2f, split_seed=%d, model_seed=%d",
                args.labeled_frac, args.split_seed, args.seed)
    X_p, X_u = load_pu_data(args.labeled_frac, args.split_seed)
    logger.info("P: %d samples, U: %d samples", len(X_p), len(X_u))

    # Validation set
    valid_data = np.load(EMBEDDING_DIR / "valid.npz", allow_pickle=False)
    X_valid, y_valid = valid_data["embeddings"], valid_data["labels"]

    n_select = int(len(X_u) * args.neg_selection_pct)
    logger.info("Will select %d reliable negatives per iteration (%.0f%% of U)",
                n_select, args.neg_selection_pct * 100)

    best_overall_f1 = -1.0
    best_overall_state = None
    model_prev = None

    for iteration in range(args.iterations):
        logger.info("=== Self-training iteration %d/%d ===", iteration + 1, args.iterations)

        if iteration == 0:
            # Initial round: treat all U as negative
            X_train = np.concatenate([X_p, X_u])
            y_train = np.concatenate([np.ones(len(X_p)), np.zeros(len(X_u))])
        else:
            # Score U with current model
            model_prev.eval()
            X_u_t = torch.tensor(X_u, dtype=torch.float32).to(device)
            with torch.no_grad():
                u_probs = model_prev.predict_proba(X_u_t).cpu().numpy()

            # Select the most confident negatives (lowest probability)
            neg_indices = np.argsort(u_probs)[:n_select]
            X_reliable_neg = X_u[neg_indices]

            logger.info(
                "  Selected %d reliable negatives (prob range: %.4f - %.4f)",
                len(neg_indices), u_probs[neg_indices[0]], u_probs[neg_indices[-1]],
            )

            # Log contamination rate (uses true labels for diagnostics only, not training)
            split_name = "frac{:.2f}_seed{}".format(args.labeled_frac, args.split_seed)
            split_dir = SPLIT_DIR / split_name
            u_true_labels = []
            with open(split_dir / "U.jsonl") as f:
                for line in f:
                    u_true_labels.append(json.loads(line).get("true_target", 0))
            u_true_labels = np.array(u_true_labels)
            false_neg_count = int(u_true_labels[neg_indices].sum())
            logger.info(
                "  Of selected negatives, %d are actually vulnerable (%.1f%% contamination)",
                false_neg_count, 100.0 * false_neg_count / len(neg_indices),
            )

            # Retrain on P + reliable negatives
            X_train = np.concatenate([X_p, X_reliable_neg])
            y_train = np.concatenate([np.ones(len(X_p)), np.zeros(len(X_reliable_neg))])

        logger.info("  Training on %d samples (%d pos, %d neg)",
                     len(y_train), int(y_train.sum()), int((y_train == 0).sum()))

        model_prev = train_mlp(
            X_train, y_train, X_valid, y_valid, device,
            hidden_dim=args.hidden_dim, dropout=args.dropout,
            lr=args.lr, epochs=args.inner_epochs, batch_size=args.batch_size,
        )

        # Check val F1
        model_prev.eval()
        with torch.no_grad():
            val_probs = model_prev.predict_proba(
                torch.tensor(X_valid, dtype=torch.float32).to(device)
            ).cpu().numpy()
        val_f1 = f1_score(y_valid, (val_probs >= 0.5).astype(int))
        logger.info("  Iteration %d val F1: %.4f", iteration + 1, val_f1)

        if val_f1 > best_overall_f1:
            best_overall_f1 = val_f1
            best_overall_state = {k: v.cpu().clone() for k, v in model_prev.state_dict().items()}

    # Save best model
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_name = "self_training_frac{:.2f}_seed{}.pt".format(args.labeled_frac, args.seed)
    ckpt = CHECKPOINT_DIR / ckpt_name
    torch.save(best_overall_state, ckpt)
    logger.info("Best val F1 across iterations: %.4f", best_overall_f1)
    logger.info("Checkpoint saved to %s", ckpt)


if __name__ == "__main__":
    main()
