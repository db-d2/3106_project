"""MLP classifier for vulnerability detection on CodeBERT embeddings."""

import torch
import torch.nn as nn


class VulnMLP(nn.Module):
    """2-layer MLP classifier operating on fixed 768-d CodeBERT embeddings."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns raw logits (no sigmoid)."""
        return self.net(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities (sigmoid applied)."""
        return torch.sigmoid(self.forward(x))
