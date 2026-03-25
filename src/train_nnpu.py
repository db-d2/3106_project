"""Train nnPU (non-negative PU) model on CodeBERT embeddings.

Risk estimator family. Requires a class prior estimate.

Usage:
    python src/train_nnpu.py --labeled-frac 0.20 --prior 0.0277 --seed 42
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from model import VulnMLP
from losses import nnpu_loss

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train nnPU model")
    parser.add_argument("--labeled-frac", type=float, default=0.20)
    parser.add_argument("--prior", type=float, default=0.0277,
                        help="Estimated class prior (default: 0.0277 from PrimeVul)")
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

    # Load data
    logger.info("Loading PU split: frac=%.2f, split_seed=%d, model_seed=%d",
                args.labeled_frac, args.split_seed, args.seed)
    X_p, X_u = load_pu_data(args.labeled_frac, args.split_seed)
    logger.info("P: %d samples, U: %d samples, prior: %.4f", len(X_p), len(X_u), args.prior)

    # Validation set (full labels, for monitoring only)
    valid_data = np.load(EMBEDDING_DIR / "valid.npz", allow_pickle=False)
    X_valid = torch.tensor(valid_data["embeddings"], dtype=torch.float32).to(device)
    y_valid = valid_data["labels"]

    # Convert to tensors
    X_p_t = torch.tensor(X_p, dtype=torch.float32)
    X_u_t = torch.tensor(X_u, dtype=torch.float32)

    p_loader = DataLoader(TensorDataset(X_p_t), batch_size=min(args.batch_size, len(X_p)), shuffle=True)
    u_loader = DataLoader(TensorDataset(X_u_t), batch_size=args.batch_size, shuffle=True)

    model = VulnMLP(input_dim=X_p.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_f1 = -1.0
    patience_counter = 0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_name = "nnpu_frac{:.2f}_prior{:.4f}_seed{}.pt".format(
        args.labeled_frac, args.prior, args.seed
    )
    ckpt = CHECKPOINT_DIR / ckpt_name

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # Iterate over U (the large set) and cycle P (the small set).
        # This ensures each epoch processes all of U. P is repeated as needed.
        p_iter = iter(p_loader)
        for (u_batch,) in u_loader:
            try:
                (p_batch,) = next(p_iter)
            except StopIteration:
                p_iter = iter(p_loader)
                (p_batch,) = next(p_iter)

            p_batch = p_batch.to(device)
            u_batch = u_batch.to(device)

            optimizer.zero_grad()
            logits_p = model(p_batch)
            logits_u = model(u_batch)
            loss = nnpu_loss(logits_p, logits_u, args.prior)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validate with F1 on full-label dev set
        model.eval()
        with torch.no_grad():
            val_probs = model.predict_proba(X_valid).cpu().numpy()
        val_preds = (val_probs >= 0.5).astype(int)
        val_f1 = f1_score(y_valid, val_preds)

        logger.info("Epoch %d/%d -- loss: %.4f, val_f1: %.4f", epoch, args.epochs, avg_loss, val_f1)

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
