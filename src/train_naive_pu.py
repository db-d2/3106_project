"""Train naive PU baseline: treat all unlabeled data as negative.

This is the lower bound for PU methods. Same BCE loss as the
supervised baseline, but with P labeled as 1 and all of U labeled as 0.

Usage:
    python src/train_naive_pu.py --labeled-frac 0.20 --seed 42
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

EMBEDDING_DIR = Path("data/processed/embeddings")
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


def load_pu_embeddings(
    labeled_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load P and U splits, map to embeddings, return (X, y) with U labeled as 0."""
    split_name = "frac" + f"{labeled_frac:.2f}" + "_seed" + str(seed)
    split_dir = SPLIT_DIR / split_name

    if not split_dir.exists():
        raise FileNotFoundError(
            "PU split not found: " + str(split_dir) + ". "
            "Run: python data/prepare_pu_splits.py --labeled-frac "
            + str(labeled_frac) + " --seed " + str(seed)
        )

    # Load the full training embeddings and build an index by idx
    train_data = np.load(EMBEDDING_DIR / "train.npz", allow_pickle=False)
    all_embeddings = train_data["embeddings"]
    all_idxs = train_data["idxs"]
    idx_to_pos = {int(idx): i for i, idx in enumerate(all_idxs)}

    # Load P and U record indices
    p_indices = []
    with open(split_dir / "P.jsonl") as f:
        for line in f:
            p_indices.append(idx_to_pos[json.loads(line)["idx"]])

    u_indices = []
    with open(split_dir / "U.jsonl") as f:
        for line in f:
            u_indices.append(idx_to_pos[json.loads(line)["idx"]])

    # Build training arrays: P gets label 1, U gets label 0
    all_indices = p_indices + u_indices
    X = all_embeddings[all_indices]
    y = np.concatenate([np.ones(len(p_indices)), np.zeros(len(u_indices))])

    return X, y


def make_dataloader(
    embeddings: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool = True
) -> DataLoader:
    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def validate(model, loader, device) -> float:
    """Validate model. Returns F1 score."""
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            probs = model.predict_proba(X_batch).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y_batch.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs >= 0.5).astype(int)
    return f1_score(labels, preds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train naive PU baseline")
    parser.add_argument("--labeled-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42, help="Model training seed")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="PU split seed (defaults to --seed)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    if args.split_seed is None:
        args.split_seed = args.seed

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load PU data
    logger.info("Loading PU split: frac=%.2f, split_seed=%d, model_seed=%d",
                args.labeled_frac, args.split_seed, args.seed)
    X_train, y_train = load_pu_embeddings(args.labeled_frac, args.split_seed)
    logger.info(
        "Train: %d samples (%d P as positive, %d U as negative)",
        len(y_train), int(y_train.sum()), int((y_train == 0).sum()),
    )

    # Load validation set (full labels)
    valid_data = np.load(EMBEDDING_DIR / "valid.npz", allow_pickle=False)
    X_valid, y_valid = valid_data["embeddings"], valid_data["labels"]

    # Class weight: P is tiny relative to U
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    logger.info("Positive weight: %.1f", pos_weight.item())

    train_loader = make_dataloader(X_train, y_train, args.batch_size)
    valid_loader = make_dataloader(X_valid, y_valid.astype(np.float32), args.batch_size, shuffle=False)

    model = VulnMLP(input_dim=X_train.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1 = -1.0
    patience_counter = 0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_name = "naive_pu_frac" + "{:.2f}".format(args.labeled_frac) + "_seed" + str(args.seed) + ".pt"
    ckpt = CHECKPOINT_DIR / ckpt_name

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_f1 = validate(model, valid_loader, device)
        logger.info("Epoch %d/%d -- train_loss: %.4f, val_f1: %.4f",
                     epoch, args.epochs, train_loss, val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    logger.info("Best val F1: %.4f", best_val_f1)
    logger.info("Checkpoint saved to %s", ckpt)


if __name__ == "__main__":
    main()
