"""Time-dependent MLP that estimates the score ∇_x log p_t(x)."""

from __future__ import annotations

import torch
from torch import nn

from src.diffusion.embeddings import ConditionEmbedding, SinusoidalTimeEmbedding
from src.utils.exceptions import TabDiffuseError


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = self.fc1(self.norm(hidden))
        residual = self.dropout(torch.nn.functional.silu(residual))
        residual = self.fc2(residual)
        return hidden + residual


class ScoreNetwork(nn.Module):
    """Estimate the score of noised tabular features.

    Inputs are a continuous feature vector and a diffusion time. An optional
    condition vector is added through a separate embedding so conditioning is
    not implemented by treating category codes as numeric targets.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
        time_dim: int = 128,
        cond_dim: int = 0,
    ) -> None:
        super().__init__()
        if feature_dim < 1:
            raise TabDiffuseError("The score network needs at least one feature.")
        if hidden_dim < 8 or n_layers < 1:
            raise TabDiffuseError("Hidden dimension and layer count are too small to train.")
        self.feature_dim = feature_dim
        self.cond_dim = cond_dim
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_projection = nn.Linear(time_dim, hidden_dim)
        self.condition_embedding = ConditionEmbedding(cond_dim, hidden_dim)
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(n_layers)])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, feature_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.feature_dim:
            raise TabDiffuseError(
                f"Score network expected features of width {self.feature_dim}."
            )
        hidden = self.input_projection(x)
        hidden = hidden + self.time_projection(self.time_embedding(t))
        hidden = hidden + self.condition_embedding(cond, x)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(torch.nn.functional.silu(self.output_norm(hidden)))
