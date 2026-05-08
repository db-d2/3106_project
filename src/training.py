"""Shared training functions for notebooks.

Each function trains a model and returns (model, history) where history
is a dict of per-epoch metrics for plotting learning curves.

All functions use AUROC for early stopping and return the best model.
"""

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from model import VulnMLP
from losses import nnpu_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split_embeddings(embedding_dir: str, split: str) -> dict:
    path = Path(embedding_dir) / (split + ".npz")
    data = np.load(path, allow_pickle=False)
    return {
        "embeddings": data["embeddings"],
        "labels": data["labels"],
        "cwe_types": data["cwe_types"],
        "idxs": data["idxs"],
    }


def load_pu_data(
    embedding_dir: str, split_dir: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load P and U embeddings separately from a PU split."""
    train_data = np.load(Path(embedding_dir) / "train.npz", allow_pickle=False)
    all_embeddings = train_data["embeddings"]
    all_idxs = train_data["idxs"]
    idx_to_pos = {int(idx): i for i, idx in enumerate(all_idxs)}

    p_indices = []
    with open(Path(split_dir) / "P.jsonl") as f:
        for line in f:
            p_indices.append(idx_to_pos[json.loads(line)["idx"]])

    u_indices = []
    with open(Path(split_dir) / "U.jsonl") as f:
        for line in f:
            u_indices.append(idx_to_pos[json.loads(line)["idx"]])

    return all_embeddings[p_indices], all_embeddings[u_indices]


def _val_auroc(model, X_val_t, y_valid, device):
    """Compute validation AUROC."""
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(X_val_t).cpu().numpy()
    return roc_auc_score(y_valid, probs)


def train_supervised(
    embedding_dir: str = "data/processed/embeddings",
    seed: int = 42,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    patience: int = 5,
    device: Optional[torch.device] = None,
) -> tuple[VulnMLP, dict]:
    """Train supervised MLP on full labels. Returns (model, history)."""
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = load_split_embeddings(embedding_dir, "train")
    valid = load_split_embeddings(embedding_dir, "valid")
    X_train, y_train = train["embeddings"], train["labels"]
    X_valid, y_valid = valid["embeddings"], valid["labels"]

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                       torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True)
    X_val_t = torch.tensor(X_valid, dtype=torch.float32).to(device)

    model = VulnMLP(input_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = {"train_loss": [], "val_auroc": [], "best_epoch": 0}
    best_auroc = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        val_auroc = _val_auroc(model, X_val_t, y_valid, device)
        history["train_loss"].append(epoch_loss / n_batches)
        history["val_auroc"].append(val_auroc)

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.to(device)
    return model, history


def train_naive_pu(
    embedding_dir: str = "data/processed/embeddings",
    labeled_frac: float = 0.20,
    split_seed: int = 42,
    model_seed: int = 42,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    patience: int = 5,
    device: Optional[torch.device] = None,
) -> tuple[VulnMLP, dict]:
    """Train naive PU (treat U as negative). Returns (model, history)."""
    set_seed(model_seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_dir = str(Path(embedding_dir).parent / "frac{:.2f}_seed{}".format(labeled_frac, split_seed))
    X_p, X_u = load_pu_data(embedding_dir, split_dir)
    X_train = np.concatenate([X_p, X_u])
    y_train = np.concatenate([np.ones(len(X_p)), np.zeros(len(X_u))])

    valid = load_split_embeddings(embedding_dir, "valid")
    X_valid, y_valid = valid["embeddings"], valid["labels"]

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                       torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True)
    X_val_t = torch.tensor(X_valid, dtype=torch.float32).to(device)

    model = VulnMLP(input_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = {"train_loss": [], "val_auroc": [], "best_epoch": 0}
    best_auroc = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        val_auroc = _val_auroc(model, X_val_t, y_valid, device)
        history["train_loss"].append(epoch_loss / n_batches)
        history["val_auroc"].append(val_auroc)

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.to(device)
    return model, history


def train_nnpu(
    embedding_dir: str = "data/processed/embeddings",
    labeled_frac: float = 0.20,
    split_seed: int = 42,
    model_seed: int = 42,
    prior: float = 0.0277,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    patience: int = 5,
    device: Optional[torch.device] = None,
) -> tuple[VulnMLP, dict]:
    """Train nnPU (risk estimator). Returns (model, history) with loss decomposition."""
    set_seed(model_seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_dir = str(Path(embedding_dir).parent / "frac{:.2f}_seed{}".format(labeled_frac, split_seed))
    X_p, X_u = load_pu_data(embedding_dir, split_dir)

    valid = load_split_embeddings(embedding_dir, "valid")
    X_valid, y_valid = valid["embeddings"], valid["labels"]
    X_val_t = torch.tensor(X_valid, dtype=torch.float32).to(device)

    X_p_t = torch.tensor(X_p, dtype=torch.float32)
    X_u_t = torch.tensor(X_u, dtype=torch.float32)
    p_loader = DataLoader(TensorDataset(X_p_t), batch_size=min(batch_size, len(X_p)), shuffle=True)
    u_loader = DataLoader(TensorDataset(X_u_t), batch_size=batch_size, shuffle=True)

    model = VulnMLP(input_dim=X_p.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [], "val_auroc": [],
        "positive_risk": [], "negative_risk": [], "neg_clipped_count": [],
        "best_epoch": 0,
    }
    best_auroc = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_pos_risk = 0.0
        epoch_neg_risk = 0.0
        epoch_clips = 0
        n_batches = 0

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

            # Track loss components
            loss_p_pos = F.binary_cross_entropy_with_logits(
                logits_p, torch.ones_like(logits_p), reduction="mean")
            loss_p_neg = F.binary_cross_entropy_with_logits(
                logits_p, torch.zeros_like(logits_p), reduction="mean")
            loss_u_neg = F.binary_cross_entropy_with_logits(
                logits_u, torch.zeros_like(logits_u), reduction="mean")
            pos_risk_val = (prior * loss_p_pos).item()
            neg_risk_val = (loss_u_neg - prior * loss_p_neg).item()
            clipped = neg_risk_val < 0

            loss = nnpu_loss(logits_p, logits_u, prior)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_pos_risk += pos_risk_val
            epoch_neg_risk += neg_risk_val
            epoch_clips += int(clipped)
            n_batches += 1

        val_auroc = _val_auroc(model, X_val_t, y_valid, device)

        history["train_loss"].append(epoch_loss / n_batches)
        history["val_auroc"].append(val_auroc)
        history["positive_risk"].append(epoch_pos_risk / n_batches)
        history["negative_risk"].append(epoch_neg_risk / n_batches)
        history["neg_clipped_count"].append(epoch_clips)

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.to(device)
    return model, history


def train_self_training(
    embedding_dir: str = "data/processed/embeddings",
    labeled_frac: float = 0.20,
    split_seed: int = 42,
    model_seed: int = 42,
    iterations: int = 5,
    neg_selection_pct: float = 0.10,
    inner_epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    device: Optional[torch.device] = None,
) -> tuple[VulnMLP, dict]:
    """Train self-training (pseudo-labeling). Returns (model, history) with per-iteration tracking."""
    set_seed(model_seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_dir = str(Path(embedding_dir).parent / "frac{:.2f}_seed{}".format(labeled_frac, split_seed))
    X_p, X_u = load_pu_data(embedding_dir, split_dir)

    valid = load_split_embeddings(embedding_dir, "valid")
    X_valid, y_valid = valid["embeddings"], valid["labels"]
    X_val_t = torch.tensor(X_valid, dtype=torch.float32).to(device)

    # Load true targets for contamination tracking
    u_true_labels = []
    with open(Path(split_dir) / "U.jsonl") as f:
        for line in f:
            u_true_labels.append(json.loads(line).get("true_target", 0))
    u_true_labels = np.array(u_true_labels)

    n_select = int(len(X_u) * neg_selection_pct)

    history = {
        "iter_val_f1": [], "iter_val_auroc": [],
        "iter_n_selected": [], "iter_contamination_rate": [],
        "iter_prob_range": [],
        "best_iteration": 0,
    }
    best_overall_auroc = -1.0
    best_overall_state = None
    model_prev = None

    for iteration in range(iterations):
        if iteration == 0:
            X_train = np.concatenate([X_p, X_u])
            y_train = np.concatenate([np.ones(len(X_p)), np.zeros(len(X_u))])
            history["iter_n_selected"].append(len(X_u))
            history["iter_contamination_rate"].append(
                float(u_true_labels.sum()) / len(u_true_labels))
            history["iter_prob_range"].append((0.0, 1.0))
        else:
            model_prev.eval()
            X_u_t = torch.tensor(X_u, dtype=torch.float32).to(device)
            with torch.no_grad():
                u_probs = model_prev.predict_proba(X_u_t).cpu().numpy()

            neg_indices = np.argsort(u_probs)[:n_select]
            X_reliable_neg = X_u[neg_indices]

            contamination = float(u_true_labels[neg_indices].sum()) / len(neg_indices)
            prob_range = (float(u_probs[neg_indices[0]]), float(u_probs[neg_indices[-1]]))
            history["iter_n_selected"].append(len(neg_indices))
            history["iter_contamination_rate"].append(contamination)
            history["iter_prob_range"].append(prob_range)

            X_train = np.concatenate([X_p, X_reliable_neg])
            y_train = np.concatenate([np.ones(len(X_p)), np.zeros(len(X_reliable_neg))])

        # Train inner MLP
        n_pos = max(int(y_train.sum()), 1)
        n_neg = len(y_train) - n_pos
        pw = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                           torch.tensor(y_train, dtype=torch.float32)),
            batch_size=batch_size, shuffle=True)

        model_prev = VulnMLP(input_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
        opt = torch.optim.Adam(model_prev.parameters(), lr=lr)

        inner_best_auroc = -1.0
        inner_best_state = None
        inner_wait = 0
        for _ in range(inner_epochs):
            model_prev.train()
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                opt.zero_grad()
                criterion(model_prev(X_b), y_b).backward()
                opt.step()

            auroc = _val_auroc(model_prev, X_val_t, y_valid, device)
            if auroc > inner_best_auroc:
                inner_best_auroc = auroc
                inner_best_state = {k: v.cpu().clone() for k, v in model_prev.state_dict().items()}
                inner_wait = 0
            else:
                inner_wait += 1
                if inner_wait >= 5:
                    break

        model_prev.load_state_dict(inner_best_state)
        model_prev.to(device)

        val_auroc = _val_auroc(model_prev, X_val_t, y_valid, device)
        val_probs = model_prev.predict_proba(X_val_t).cpu().detach().numpy()
        val_f1 = f1_score(y_valid, (val_probs >= 0.5).astype(int))

        history["iter_val_f1"].append(val_f1)
        history["iter_val_auroc"].append(val_auroc)

        if val_auroc > best_overall_auroc:
            best_overall_auroc = val_auroc
            best_overall_state = {k: v.cpu().clone() for k, v in model_prev.state_dict().items()}
            history["best_iteration"] = iteration

    model_prev.load_state_dict(best_overall_state)
    model_prev.to(device)
    return model_prev, history


def find_best_threshold(
    model: VulnMLP,
    X_val: torch.Tensor,
    y_val: np.ndarray,
) -> tuple[float, float]:
    """Find threshold maximizing F1 on validation data."""
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(X_val).cpu().numpy()

    best_f1 = -1.0
    best_thr = 0.5
    for thr in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y_val, (probs >= thr).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1
