"""Gradient clipping keeps parameter updates bounded (torch extra)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.model


def test_clip_bounds_total_grad_norm() -> None:
    torch = pytest.importorskip("torch")
    from wavrail.model.network import AcousticModel

    model = AcousticModel(input_dim=6, hidden_dim=8, vocab_size=4)
    torch.manual_seed(7)
    x = torch.randn(2, 5, 6)
    logits = model(x)
    loss = logits.square().mean()
    loss.backward()
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
    assert float(total) > 0.01  # clipping actually triggered
    after = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    assert float(after) <= 0.01 + 1e-4


def test_trainer_runs_with_tiny_clip() -> None:
    from wavrail.model import TrainingConfig, build_synthetic_dataset, train_synthetic
    from wavrail.vocab import Vocab

    vocab = Vocab.from_labels(["<blank>", "a", "b", "c"])

    def factory(seed: int) -> tuple[list, list]:
        examples = build_synthetic_dataset(vocab, n_examples=4, duration_s=0.2, seed=seed)
        return examples[:3], examples[3:]

    config = TrainingConfig(n_epochs=2, batch_size=2, hidden_dim=8, grad_clip=1e-4)
    report = train_synthetic(config, factory)
    assert len(report.epoch_losses) == 2
    assert all(loss == loss for loss in report.epoch_losses)  # no NaNs
