"""Learning-rate schedule behaviour of the trainer (torch extra)."""

from __future__ import annotations

import pytest

from wavrail.model import TrainingConfig, build_synthetic_dataset, train_synthetic
from wavrail.vocab import Vocab

pytestmark = pytest.mark.model


def _factory(seed: int) -> tuple[list, list]:
    vocab = Vocab.from_labels(["<blank>", "a", "b", "c"])
    examples = build_synthetic_dataset(vocab, n_examples=4, duration_s=0.2, seed=seed)
    return examples[:3], examples[3:]


def test_learning_rates_recorded_per_epoch() -> None:
    config = TrainingConfig(n_epochs=3, batch_size=2, hidden_dim=8)
    report = train_synthetic(config, _factory)
    assert report.learning_rates == [config.learning_rate] * 3


def test_step_decay_halves_every_two_epochs() -> None:
    config = TrainingConfig(
        n_epochs=5,
        batch_size=2,
        hidden_dim=8,
        learning_rate=1e-2,
        lr_decay_factor=0.5,
        lr_decay_every=2,
    )
    report = train_synthetic(config, _factory)
    assert report.learning_rates == pytest.approx([1e-2, 1e-2, 5e-3, 5e-3, 2.5e-3])


def test_loss_still_decreases_with_decay() -> None:
    config = TrainingConfig(
        n_epochs=4,
        batch_size=2,
        learning_rate=5e-3,
        hidden_dim=16,
        lr_decay_factor=0.7,
        lr_decay_every=2,
    )
    report = train_synthetic(config, _factory)
    assert report.loss_decreased()


def test_rejects_invalid_decay_settings() -> None:
    from wavrail.errors import ModelError

    with pytest.raises(ModelError):
        TrainingConfig(lr_decay_factor=0.0)
    with pytest.raises(ModelError):
        TrainingConfig(lr_decay_every=-1)
