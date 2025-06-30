"""Held-out dev CER reporting (requires the torch extra)."""

from __future__ import annotations

import pytest

from wavrail.model import TrainingConfig, build_synthetic_dataset, train_synthetic
from wavrail.vocab import Vocab

pytestmark = pytest.mark.model


def _vocab() -> Vocab:
    return Vocab.from_labels(["<blank>", "a", "b", "c"])


def _factory(seed: int) -> tuple[list, list]:
    examples = build_synthetic_dataset(_vocab(), n_examples=4, duration_s=0.2, seed=seed)
    return examples[:3], examples[3:]


def test_dev_cer_is_reported_with_dataset_vocab() -> None:
    config = TrainingConfig(n_epochs=6, batch_size=2, learning_rate=5e-3, hidden_dim=16, seed=3)
    report = train_synthetic(config, _factory, vocab=_vocab())
    assert report.dev_cer is not None
    assert report.dev_cer >= 0.0
    assert report.loss_decreased()


def test_dev_cer_none_without_dev_split() -> None:
    vocab = _vocab()

    def factory(seed: int) -> tuple[list, list]:
        examples = build_synthetic_dataset(vocab, n_examples=3, duration_s=0.2, seed=seed)
        return examples, []

    config = TrainingConfig(n_epochs=2, batch_size=2, hidden_dim=8, seed=1)
    report = train_synthetic(config, factory, vocab=vocab)
    assert report.dev_cer is None


def test_dev_cer_still_computed_without_vocab() -> None:
    # without the dataset vocab the fallback label set still yields a
    # finite, non-negative rate; it is a health signal, not accuracy
    config = TrainingConfig(n_epochs=2, batch_size=2, hidden_dim=8, seed=1)
    report = train_synthetic(config, _factory)
    assert report.dev_cer is not None
    assert report.dev_cer >= 0.0
