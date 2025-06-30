"""Validation of TrainingConfig hyperparameters (no torch required)."""

from __future__ import annotations

import pytest

from wavrail.errors import ModelError
from wavrail.model import TrainingConfig


def test_defaults_are_valid() -> None:
    config = TrainingConfig()
    assert config.n_epochs == 4
    assert config.grad_clip == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_epochs": -1},
        {"batch_size": 0},
        {"learning_rate": 0.0},
        {"learning_rate": -1e-3},
        {"grad_clip": 0.0},
        {"hidden_dim": 0},
    ],
)
def test_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises(ModelError):
        TrainingConfig(**kwargs)
