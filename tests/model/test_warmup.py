"""Learning-rate warmup and its interaction with step decay (torch extra)."""

from __future__ import annotations

import pytest

from wavrail.errors import ModelError
from wavrail.model import TrainingConfig, build_synthetic_dataset, train_synthetic
from wavrail.vocab import Vocab

pytestmark = pytest.mark.model


def _factory(seed: int) -> tuple[list, list]:
    vocab = Vocab.from_labels(["<blank>", "a", "b", "c"])
    examples = build_synthetic_dataset(vocab, n_examples=4, duration_s=0.2, seed=seed)
    return examples[:3], examples[3:]


def test_warmup_ramps_linearly() -> None:
    config = TrainingConfig(
        n_epochs=4, batch_size=2, hidden_dim=8, learning_rate=1e-2, lr_warmup_epochs=2
    )
    report = train_synthetic(config, _factory)
    assert report.learning_rates == pytest.approx([5e-3, 1e-2, 1e-2, 1e-2])


def test_decay_counts_from_end_of_warmup() -> None:
    config = TrainingConfig(
        n_epochs=6,
        batch_size=2,
        hidden_dim=8,
        learning_rate=1e-2,
        lr_warmup_epochs=2,
        lr_decay_factor=0.5,
        lr_decay_every=2,
    )
    report = train_synthetic(config, _factory)
    assert report.learning_rates == pytest.approx([5e-3, 1e-2, 1e-2, 1e-2, 5e-3, 5e-3])


def test_warmup_validation() -> None:
    with pytest.raises(ModelError):
        TrainingConfig(lr_warmup_epochs=-1)
