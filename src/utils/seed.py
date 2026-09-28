"""Reproducible seeding for Python, NumPy, and PyTorch."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed the random number generators used by training and generation."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
