"""Score-based diffusion model wrapping the schedule and score network."""

from __future__ import annotations

import torch
from torch import nn

from src.diffusion.noise_schedule import VariancePreservingSchedule
from src.diffusion.score_network import ScoreNetwork
from src.utils.exceptions import TabDiffuseError


class ScoreBasedDiffusion(nn.Module):
    """Learn ∇_x log p_t(x) for mixed tabular features in continuous space."""

    def __init__(
        self,
        feature_dim: int,
        schedule: VariancePreservingSchedule | None = None,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
        time_dim: int = 128,
        cond_dim: int = 0,
    ) -> None:
        super().__init__()
        self.schedule = schedule or VariancePreservingSchedule()
        self.score_network = ScoreNetwork(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            time_dim=time_dim,
            cond_dim=cond_dim,
        )
        self.feature_dim = feature_dim
        self.cond_dim = cond_dim

    def perturb(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw x_t from the Gaussian perturbation kernel and return the noise."""
        mean_coeff, std = self.schedule.marginal_prob(t)
        noise = torch.randn_like(x0)
        while mean_coeff.ndim < x0.ndim:
            mean_coeff = mean_coeff.unsqueeze(-1)
            std = std.unsqueeze(-1)
        x_t = mean_coeff * x0 + std * noise
        return x_t, noise, mean_coeff.squeeze(-1), std.squeeze(-1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        return self.score_network(x_t, t, cond)

    def sample_times(self, batch: int, steps: int, mode: str, device: torch.device) -> torch.Tensor:
        """Sample diffusion times, either continuously or on a uniform grid."""
        if steps < 2:
            raise TabDiffuseError("Diffusion steps must be at least 2.")
        if mode == "continuous":
            low = self.schedule.t_epsilon
            return low + (1.0 - low) * torch.rand(batch, device=device)
        if mode == "discrete":
            indices = torch.randint(1, steps + 1, (batch,), device=device)
            return indices.float() / float(steps)
        raise TabDiffuseError("Time sampling must be 'continuous' or 'discrete'.")


def build_model(feature_dim: int, config: dict, cond_dim: int = 0) -> ScoreBasedDiffusion:
    """Construct a score model from a flat configuration dictionary."""
    if str(config.get("sde", "vp")).lower() not in {"vp", "variance_preserving"}:
        raise TabDiffuseError("Only the variance-preserving SDE is implemented.")
    schedule = VariancePreservingSchedule(
        beta_min=float(config.get("beta_min", 0.1)),
        beta_max=float(config.get("beta_max", 20.0)),
        schedule=str(config.get("schedule", "linear")),
        t_epsilon=float(config.get("t_epsilon", 1e-3)),
    )
    return ScoreBasedDiffusion(
        feature_dim=feature_dim,
        schedule=schedule,
        hidden_dim=int(config.get("hidden_dim", 256)),
        n_layers=int(config.get("layers", 4)),
        dropout=float(config.get("dropout", 0.1)),
        time_dim=int(config.get("time_embedding_dim", 128)),
        cond_dim=cond_dim,
    )
