"""Variance-preserving noise schedule for a continuous SDE.

The forward process is the variance-preserving SDE

    dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dW,  t in (0, 1].

For a linear or cosine beta schedule the integrated drift has a closed form,
so the perturbation kernel p(x_t | x_0) is Gaussian with

    mean = x_0 * exp(-0.5 * ∫_0^t beta(s) ds)
    variance = 1 - exp(-∫_0^t beta(s) ds).
"""

from __future__ import annotations

import math

import torch

from src.utils.exceptions import TabDiffuseError


class VariancePreservingSchedule:
    """Closed-form marginals for the variance-preserving SDE."""

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        schedule: str = "linear",
        t_epsilon: float = 1e-3,
    ) -> None:
        if schedule not in {"linear", "cosine"}:
            raise TabDiffuseError("Noise schedule must be 'linear' or 'cosine'.")
        if beta_min <= 0 or beta_max <= beta_min:
            raise TabDiffuseError("Require 0 < beta_min < beta_max.")
        if not 0 < t_epsilon < 0.2:
            raise TabDiffuseError("t_epsilon must lie strictly between 0 and 0.2.")
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.schedule = schedule
        self.t_epsilon = float(t_epsilon)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        time = self._as_time(t)
        beta_min, delta = self._bounds(time)
        if self.schedule == "linear":
            return beta_min + time * delta
        return beta_min + 0.5 * delta * (1.0 - torch.cos(math.pi * time))

    def integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Analytical integral of beta from 0 to t."""
        time = self._as_time(t)
        beta_min, delta = self._bounds(time)
        if self.schedule == "linear":
            return beta_min * time + 0.5 * delta * time.square()
        return beta_min * time + 0.5 * delta * time - 0.5 * delta * torch.sin(math.pi * time) / math.pi

    def marginal_prob(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the mean coefficient and standard deviation of p(x_t | x_0)."""
        integrated = self.integral_beta(t)
        mean_coeff = torch.exp(-0.5 * integrated)
        variance = (1.0 - torch.exp(-integrated)).clamp_min(1e-8)
        return mean_coeff, torch.sqrt(variance)

    def _bounds(self, time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        beta_min = torch.tensor(self.beta_min, device=time.device, dtype=time.dtype)
        beta_max = torch.tensor(self.beta_max, device=time.device, dtype=time.dtype)
        return beta_min, beta_max - beta_min

    @staticmethod
    def _as_time(t: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(t, dtype=torch.float32)
