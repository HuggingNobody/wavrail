"""TrainingReport serialisation is JSON-friendly and lossless."""

from __future__ import annotations

import json

from wavrail.model import TrainingReport


def test_roundtrip_preserves_values() -> None:
    report = TrainingReport(
        epoch_losses=[1.5, 0.75, 0.5],
        learning_rates=[1e-3, 1e-3, 5e-4],
        resumed_from_epoch=1,
        final_loss=0.5,
        initial_loss=1.5,
        n_train_examples=3,
        n_dev_examples=1,
        dev_cer=0.25,
    )
    payload = report.to_dict()
    restored = TrainingReport.from_dict(json.loads(json.dumps(payload)))
    assert restored == report
    assert restored.loss_decreased()


def test_dev_cer_none_roundtrips() -> None:
    report = TrainingReport(initial_loss=1.0, final_loss=1.0)
    restored = TrainingReport.from_dict(report.to_dict())
    assert restored.dev_cer is None
    assert not restored.loss_decreased()


def test_to_dict_keys_are_stable() -> None:
    assert set(TrainingReport().to_dict()) == {
        "epoch_losses",
        "learning_rates",
        "resumed_from_epoch",
        "final_loss",
        "initial_loss",
        "n_train_examples",
        "n_dev_examples",
        "dev_cer",
        "loss_decreased",
    }
