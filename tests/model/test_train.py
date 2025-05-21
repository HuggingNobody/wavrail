"""Tests for the synthetic-model trainer (requires the torch extra)."""

from __future__ import annotations

import pytest

from wavrail.model import (
    TrainingConfig,
    build_synthetic_dataset,
    train_synthetic,
)
from wavrail.vocab import Vocab

pytestmark = pytest.mark.model


def _factory(seed: int) -> tuple[list, list]:
    vocab = Vocab.from_labels(["<blank>", "a", "b", "c"])
    examples = build_synthetic_dataset(vocab, n_examples=4, duration_s=0.2, seed=seed)
    return examples[:3], examples[3:]


def test_training_loss_decreases() -> None:
    config = TrainingConfig(n_epochs=4, batch_size=2, learning_rate=5e-3, hidden_dim=16)
    report = train_synthetic(config, _factory)
    assert report.loss_decreased()
    assert report.initial_loss > 0.0
    assert report.final_loss >= 0.0


def test_training_is_deterministic() -> None:
    config = TrainingConfig(n_epochs=2, batch_size=2, learning_rate=5e-3, hidden_dim=16, seed=42)
    a = train_synthetic(config, _factory)
    b = train_synthetic(config, _factory)
    assert a.epoch_losses == b.epoch_losses


def test_training_reports_dataset_size() -> None:
    config = TrainingConfig(n_epochs=1, batch_size=1, hidden_dim=8)
    report = train_synthetic(config, _factory)
    assert report.n_train_examples == 3
    assert report.n_dev_examples == 1
