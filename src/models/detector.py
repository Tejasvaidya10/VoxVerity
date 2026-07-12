"""
Classifier head on top of cached wav2vec2 embeddings.

Two options included:
  - LinearHead: fast baseline, embedding -> MLP -> binary logit
  - MHFAHead: multi-head factorized attention pooling + classifier, closer to
    the architecture used in recent ASVspoof-winning systems, useful if you
    switch to frame-level (unpooled) embeddings instead of mean-pooled ones
"""

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, embedding_dim) -> returns raw logit, shape (batch,)
        return self.net(x).squeeze(-1)


class MHFAHead(nn.Module):
    """Multi-head factorized attention pooling over frame-level SSL features,
    followed by a classifier. Use this if you extract unpooled hidden states
    (time, dim) per clip instead of mean-pooling in extract_features.py."""

    def __init__(self, embedding_dim: int, num_heads: int = 32,
                 compression_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.compression_dim = compression_dim
        self.key_proj = nn.Linear(embedding_dim, num_heads)
        self.value_proj = nn.Linear(embedding_dim, compression_dim)
        self.classifier = nn.Sequential(
            nn.Linear(compression_dim * num_heads, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, embedding_dim)
        attn_logits = self.key_proj(x)                       # (batch, time, num_heads)
        attn_weights = torch.softmax(attn_logits, dim=1)      # softmax over time
        values = self.value_proj(x)                           # (batch, time, compression_dim)

        # weighted sum per head: (batch, num_heads, compression_dim)
        pooled = torch.einsum("bth,btc->bhc", attn_weights, values)
        pooled = pooled.reshape(pooled.size(0), -1)            # (batch, num_heads * compression_dim)
        return self.classifier(pooled).squeeze(-1)


def build_head(config: dict) -> nn.Module:
    return LinearHead(
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    )
