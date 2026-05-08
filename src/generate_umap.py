"""Generate UMAP visualization of CodeBERT embeddings colored by label.

Subsamples all vulnerable + 10K random benign functions, fits UMAP,
and saves a scatter plot to figures/umap_embeddings.png.

Usage:
    python src/generate_umap.py
"""

import numpy as np
import umap
import matplotlib.pyplot as plt
from pathlib import Path

EMBEDDING_FILE = Path("data/processed/embeddings/train.npz")
OUTPUT_FILE = Path("figures/umap_embeddings.png")


def main():
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#e6edf3",
        "text.color": "#e6edf3",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "font.size": 12,
    })

    data = np.load(EMBEDDING_FILE, allow_pickle=False)
    X = data["embeddings"]
    y = data["labels"]
    print("Loaded {} embeddings".format(X.shape[0]))

    # Subsample: all positives + 10K random negatives
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.RandomState(42)
    neg_sample = rng.choice(neg_idx, size=10000, replace=False)
    idx = np.concatenate([neg_sample, pos_idx])
    rng.shuffle(idx)

    X_sub = X[idx]
    y_sub = y[idx]
    print("Subsampled to {} ({} pos, {} neg)".format(
        len(X_sub), int(y_sub.sum()), int((y_sub == 0).sum())))

    print("Fitting UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(10, 8))

    neg_mask = y_sub == 0
    pos_mask = y_sub == 1

    ax.scatter(embedding[neg_mask, 0], embedding[neg_mask, 1],
               c="#8b949e", s=6, alpha=0.5,
               label="Benign (n=10,000)", rasterized=True)
    ax.scatter(embedding[pos_mask, 0], embedding[pos_mask, 1],
               c="#f85149", s=8, alpha=0.35,
               label="Vulnerable (n=4,862)", rasterized=True)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("CodeBERT [CLS] embeddings: vulnerable vs benign functions",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11, framealpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved {}".format(OUTPUT_FILE))


if __name__ == "__main__":
    main()
