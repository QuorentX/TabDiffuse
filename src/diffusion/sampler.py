"""Reverse-time sampling for the variance-preserving SDE.

Reverse SDE, with time running backward:

    dx = [f(x, t) - g(t)^2 score] dt + g(t) dW_bar

where f = -0.5 beta(t) x and g = sqrt(beta(t)). The probability-flow ODE drops
the stochastic term and uses half the score correction.

Known coordinates, when a condition is requested, are re-injected at the
current noise level after each step (a RePaint-style update) and set to the
clean encoded value at the end.

Each step predicts the clean encoded row from the score and keeps that
prediction inside the training feature range. The previous Euler update
multiplied the latent when the score was still near zero, which sent decoded
ages, amounts, and dates far outside the real table.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from src.diffusion.model import ScoreBasedDiffusion
from src.utils.exceptions import TabDiffuseError


@torch.no_grad()
def sample_latent(
    model: ScoreBasedDiffusion,
    n_samples: int,
    *,
    steps: int,
    sampler: str = "reverse_sde",
    temperature: float = 1.0,
    condition: torch.Tensor | None = None,
    condition_indices: list[int] | None = None,
    condition_values: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    callback: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    """Draw samples in the preprocessor's continuous feature space."""
    if n_samples < 1:
        raise TabDiffuseError("The number of synthetic records must be at least 1.")
    if steps < 2:
        raise TabDiffuseError("Sampling steps must be at least 2.")
    if sampler not in {"reverse_sde", "probability_flow"}:
        raise TabDiffuseError("Sampler must be 'reverse_sde' or 'probability_flow'.")
    if temperature <= 0:
        raise TabDiffuseError("Temperature must be positive.")

    device = next(model.parameters()).device
    model.eval()
    x = torch.randn(n_samples, model.feature_dim, device=device)
    epsilon = model.schedule.t_epsilon
    times = torch.linspace(1.0, epsilon, steps + 1, device=device)
    low, high = _feature_bounds(model, device)
    cond = None
    if condition is not None:
        cond = condition.to(device)
        if cond.shape[0] == 1 and n_samples > 1:
            cond = cond.expand(n_samples, -1)
        if cond.shape[0] != n_samples:
            raise TabDiffuseError("The condition batch does not match the number of samples.")

    for step in range(steps):
        t_curr = times[step]
        t_next = times[step + 1]
        t_batch = torch.full((n_samples,), float(t_curr), device=device)
        score = _score(model, x, t_batch, cond, guidance_scale)
        x0 = _predict_clean(model, x, t_batch, score)
        x0 = _clip_features(x0, low, high)
        next_batch = torch.full((n_samples,), float(t_next), device=device)
        mean_next, std_next = model.schedule.marginal_prob(next_batch)
        mean_next = mean_next.view(-1, 1)
        std_next = std_next.view(-1, 1)
        if sampler == "reverse_sde":
            x = mean_next * x0 + std_next * float(temperature) * torch.randn_like(x)
        else:
            mean_curr, std_curr = model.schedule.marginal_prob(t_batch)
            z_hat = (x - mean_curr.view(-1, 1) * x0) / std_curr.view(-1, 1).clamp_min(1e-4)
            x = mean_next * x0 + std_next * z_hat
        if condition_indices and condition_values is not None:
            mean_coeff, std = model.schedule.marginal_prob(t_next)
            clean = condition_values.to(device)
            if clean.shape[0] == 1 and n_samples > 1:
                clean = clean.expand(n_samples, -1)
            injected_noise = torch.randn_like(clean) * float(temperature)
            injected = mean_coeff * clean + std * injected_noise
            x[:, condition_indices] = injected
        if callback is not None:
            callback(step + 1, steps)

    t_final = torch.full((n_samples,), epsilon, device=device)
    score = _score(model, x, t_final, cond, guidance_scale)
    x = _clip_features(_predict_clean(model, x, t_final, score), low, high)
    if condition_indices and condition_values is not None:
        clean = condition_values.to(device)
        if clean.shape[0] == 1 and n_samples > 1:
            clean = clean.expand(n_samples, -1)
        x[:, condition_indices] = clean
    return x


def _score(
    model: ScoreBasedDiffusion,
    x: torch.Tensor,
    t_batch: torch.Tensor,
    cond: torch.Tensor | None,
    guidance_scale: float,
) -> torch.Tensor:
    if cond is None or model.cond_dim == 0 or guidance_scale == 1.0:
        return model(x, t_batch, cond if model.cond_dim else None)
    conditional = model(x, t_batch, cond)
    unconditional = model(x, t_batch, torch.zeros_like(cond))
    return unconditional + guidance_scale * (conditional - unconditional)


def _feature_bounds(model: ScoreBasedDiffusion, device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Training-encoded min and max, stored on the model by the generator."""
    low = getattr(model, "feature_low", None)
    high = getattr(model, "feature_high", None)
    if low is None or high is None:
        return None, None
    low_t = torch.as_tensor(low, dtype=torch.float32, device=device).view(1, -1)
    high_t = torch.as_tensor(high, dtype=torch.float32, device=device).view(1, -1)
    if low_t.shape[1] != model.feature_dim or high_t.shape[1] != model.feature_dim:
        return None, None
    return low_t, high_t


def _predict_clean(
    model: ScoreBasedDiffusion,
    x: torch.Tensor,
    t_batch: torch.Tensor,
    score: torch.Tensor,
) -> torch.Tensor:
    """Tweedie estimate of x0. A weak score no longer amplifies the latent."""
    mean_coeff, std = model.schedule.marginal_prob(t_batch)
    mean_coeff = mean_coeff.view(-1, 1).clamp_min(1e-4)
    std = std.view(-1, 1)
    return (x + std.square() * score) / mean_coeff


def _clip_features(x0: torch.Tensor, low: torch.Tensor | None, high: torch.Tensor | None) -> torch.Tensor:
    if low is None or high is None:
        return x0.clamp(-6.0, 6.0)
    return torch.max(torch.min(x0, high), low)
