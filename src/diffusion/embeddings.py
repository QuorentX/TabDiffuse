"""Time and conditioning embeddings for the score network."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Map a continuous diffusion time in (0, 1] to a sinusoidal vector."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 2:
            raise ValueError("Time embedding dimension must be at least 2.")
        self.embedding_dim = embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.SiLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.embedding_dim // 2
        frequencies = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1)
        )
        angles = t.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if embedding.shape[1] < self.embedding_dim:
            embedding = torch.nn.functional.pad(embedding, (0, self.embedding_dim - embedding.shape[1]))
        return self.projection(embedding)


class ConditionEmbedding(nn.Module):
    """Project a clean condition vector. An absent condition contributes zeros.

    Classifier-free training passes a zero vector for a fraction of batches so
    the same network can sample with or without a condition.
    """

    def __init__(self, cond_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        if cond_dim > 0:
            self.projection: nn.Module | None = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.projection = None

    def forward(self, cond: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
        if self.projection is None:
            return torch.zeros(reference.shape[0], self.hidden_dim, device=reference.device, dtype=reference.dtype)
        if cond is None:
            cond = torch.zeros(reference.shape[0], self.cond_dim, device=reference.device, dtype=reference.dtype)
        return self.projection(cond)
