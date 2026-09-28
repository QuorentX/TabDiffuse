"""Weighted denoising score matching loss.

The perturbation kernel score is -z / sigma_t. Weighting the squared error by
sigma_t^2 gives the stable objective

    E || sigma_t * s_theta(x_t, t) + z ||^2

used by Song et al. for score-based generative models.
"""

from __future__ import annotations

import torch


def denoising_score_matching_loss(score: torch.Tensor, noise: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Return the mean weighted denoising-score-matching loss."""
    scale = std
    if scale.ndim == 1:
        scale = scale.unsqueeze(1)
    residual = score * scale + noise
    return torch.mean(residual.square())
