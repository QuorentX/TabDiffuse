"""Score-based diffusion tests."""

from __future__ import annotations

import torch

from src.diffusion.loss import denoising_score_matching_loss
from src.diffusion.model import build_model
from src.diffusion.noise_schedule import VariancePreservingSchedule
from src.diffusion.sampler import sample_latent
from src.diffusion.trainer import DiffusionTrainer
from src.utils.seed import set_seed


def test_noise_schedule_integral_matches_beta():
    schedule = VariancePreservingSchedule(beta_min=0.1, beta_max=2.0, schedule="linear")
    time = torch.linspace(0.05, 0.95, 8)
    integrated = schedule.integral_beta(time)
    step = 1e-3
    numerical = (schedule.integral_beta(time + step) - schedule.integral_beta(time - step)) / (2 * step)
    assert torch.allclose(numerical, schedule.beta(time), atol=1e-3)
    mean, std = schedule.marginal_prob(torch.tensor([0.2, 0.8]))
    assert torch.all(mean > 0)
    assert torch.all((std > 0) & (std < 1.1))
    assert float(mean[0]) > float(mean[1])


def test_cosine_schedule_is_positive():
    schedule = VariancePreservingSchedule(schedule="cosine", beta_min=0.1, beta_max=5.0)
    beta = schedule.beta(torch.linspace(0, 1, 6))
    assert torch.all(beta > 0)


def test_score_network_shape_and_loss_backward():
    set_seed(7)
    model = build_model(4, {"hidden_dim": 32, "layers": 2, "dropout": 0.0, "steps": 20, "time_embedding_dim": 16})
    x0 = torch.randn(16, 4)
    t = model.sample_times(16, 20, "discrete", torch.device("cpu"))
    x_t, noise, _, std = model.perturb(x0, t)
    score = model(x_t, t, None)
    assert score.shape == x0.shape
    loss = denoising_score_matching_loss(score, noise, std)
    loss.backward()
    assert torch.isfinite(loss)


def test_sampling_shape():
    model = build_model(3, {"hidden_dim": 16, "layers": 1, "dropout": 0.0, "steps": 8, "time_embedding_dim": 8})
    samples = sample_latent(model, 5, steps=8, sampler="reverse_sde")
    assert samples.shape == (5, 3)
    assert torch.isfinite(samples).all()
    assert torch.max(samples.abs()) <= 6.0 + 1e-4


def test_training_reduces_loss_on_a_simple_sample():
    set_seed(11)
    model = build_model(3, {"hidden_dim": 32, "layers": 2, "dropout": 0.0, "steps": 20, "time_embedding_dim": 16})
    data = torch.randn(128, 3).numpy()
    trainer = DiffusionTrainer(
        model,
        {
            "epochs": 15,
            "batch_size": 32,
            "learning_rate": 2e-3,
            "weight_decay": 0.0,
            "steps": 20,
            "time_sampling": "continuous",
            "grad_clip": 1.0,
            "seed": 11,
            "condition_indices": [],
            "condition_dropout": 0.0,
        },
        torch.device("cpu"),
    )
    summary = trainer.fit(data[:100], data[100:])
    assert len(summary["history"]["train_loss"]) == 15
    assert summary["history"]["train_loss"][-1] < summary["history"]["train_loss"][0]
