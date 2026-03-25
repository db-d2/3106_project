"""Train supervised MLP baseline on CodeBERT embeddings with full labels.

This is the upper bound -- the best we can do with complete supervision.

Usage:
    python src/train_supervised.py --seed 42
"""

import argparse
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
CHECKPOINT_DIR = Path("checkpoints")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_embeddings(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings and labels for a split."""
    data = np.load(EMBEDDING_DIR / (split + ".npz"), allow_pickle=False)
    return data["embeddings"], data["labels"]


def make_dataloader(
    embeddings: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader from numpy arrays."""
    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(
    model: VulnMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def validate(
    model: VulnMLP,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Validate model. Returns F1 score on validation set."""
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
    parser = argparse.ArgumentParser(description="Train supervised MLP baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load data
    logger.info("Loading embeddings")
    X_train, y_train = load_embeddings("train")
    X_valid, y_valid = load_embeddings("valid")

    logger.info("Train: %d samples (%d positive)", len(y_train), int(y_train.sum()))
    logger.info("Valid: %d samples (%d positive)", len(y_valid), int(y_valid.sum()))

    # Class weight for imbalanced data
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    logger.info("Positive weight: %.1f", pos_weight.item())

    train_loader = make_dataloader(X_train, y_train, args.batch_size, shuffle=True)
    valid_loader = make_dataloader(X_valid, y_valid, args.batch_size, shuffle=False)

    # Model
    model = VulnMLP(
        input_dim=X_train.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training loop with early stopping on val F1
    best_val_f1 = -1.0
    patience_counter = 0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / ("supervised_seed" + str(args.seed) + ".pt")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_f1 = validate(model, valid_loader, device)

        logger.info(
            "Epoch %d/%d -- train_loss: %.4f, val_f1: %.4f",
            epoch, args.epochs, train_loss, val_f1,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    logger.info("Best val F1: %.4f", best_val_f1)
    logger.info("Checkpoint saved to %s", checkpoint_path)


if __name__ == "__main__":
    main()
