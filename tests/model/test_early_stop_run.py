"""Early stopping inside train_synthetic (torch extra).

The plateaued run uses a tiny learning rate whose per-epoch loss
change (~0.07) stays below ``early_stop_min_delta=0.5``, so with
``patience=2`` training must stop after exactly three epochs; the
loss curve is seeded and therefore reproducible.
"""

from __future__ import annotations

import pytest

from wavrail.model import (
    TrainingConfig,
    build_synthetic_dataset,
    load_checkpoint,
    train_synthetic,
)
from wavrail.vocab import Vocab

pytestmark = pytest.mark.model


def _factory(seed: int) -> tuple[list, list]:
    vocab = Vocab.from_labels(["<blank>", "a", "b", "c"])
    examples = build_synthetic_dataset(vocab, n_examples=4, duration_s=0.2, seed=seed)
    return examples[:3], examples[3:]


def test_plateaued_run_stops_early() -> None:
    config = TrainingConfig(
        n_epochs=20,
        batch_size=2,
        hidden_dim=4,
        learning_rate=1e-4,
        early_stop_patience=2,
        early_stop_min_delta=0.5,
    )
    report = train_synthetic(config, _factory)
    assert report.stopped_early
    assert len(report.epoch_losses) == 3


def test_checkpoint_records_stopped_epoch(tmp_path) -> None:
    ckpt = tmp_path / "early.pt"
    config = TrainingConfig(
        n_epochs=20,
        batch_size=2,
        hidden_dim=4,
        learning_rate=1e-4,
        early_stop_patience=2,
        early_stop_min_delta=0.5,
    )
    train_synthetic(config, _factory, save_to=ckpt)
    assert load_checkpoint(ckpt).epoch == 3


def test_disabled_by_default() -> None:
    config = TrainingConfig(n_epochs=4, batch_size=2, hidden_dim=8)
    report = train_synthetic(config, _factory)
    assert not report.stopped_early
    assert len(report.epoch_losses) == 4


def test_stopped_early_roundtrips_serde() -> None:
    from wavrail.model import TrainingReport

    report = TrainingReport(stopped_early=True)
    assert TrainingReport.from_dict(report.to_dict()).stopped_early
