"""Resume-from-checkpoint behaviour of the trainer (torch extra)."""

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


def _config(n_epochs: int, **kwargs) -> TrainingConfig:
    base = {"batch_size": 2, "learning_rate": 5e-3, "hidden_dim": 16, "seed": 3}
    base.update(kwargs)
    return TrainingConfig(n_epochs=n_epochs, **base)


def test_resume_matches_uninterrupted_run(tmp_path) -> None:
    ckpt = tmp_path / "ckpt.pt"
    train_synthetic(_config(2), _factory, save_to=ckpt)
    continuous = train_synthetic(_config(4), _factory)
    resumed = train_synthetic(_config(4), _factory, resume_from=ckpt)
    assert resumed.resumed_from_epoch == 2
    assert resumed.epoch_losses == pytest.approx(continuous.epoch_losses, abs=1e-12)
    assert resumed.final_loss == pytest.approx(continuous.final_loss, abs=1e-12)


def test_resume_rejects_hidden_dim_mismatch(tmp_path) -> None:
    ckpt = tmp_path / "ckpt.pt"
    train_synthetic(_config(1), _factory, save_to=ckpt)
    with pytest.raises(ModelError, match="hidden_dim"):
        train_synthetic(_config(2, hidden_dim=8), _factory, resume_from=ckpt)


def test_resume_rejects_seed_mismatch(tmp_path) -> None:
    ckpt = tmp_path / "ckpt.pt"
    train_synthetic(_config(1), _factory, save_to=ckpt)
    with pytest.raises(ModelError, match="seed"):
        train_synthetic(_config(2, seed=9), _factory, resume_from=ckpt)


def test_save_to_creates_parent_directories(tmp_path) -> None:
    ckpt = tmp_path / "nested" / "dir" / "ckpt.pt"
    train_synthetic(_config(1, hidden_dim=8), _factory, save_to=ckpt)
    assert ckpt.is_file()


def test_resume_past_saved_epoch_is_noop(tmp_path) -> None:
    ckpt = tmp_path / "ckpt.pt"
    first = train_synthetic(_config(2), _factory, save_to=ckpt)
    resumed = train_synthetic(_config(2), _factory, resume_from=ckpt)
    assert resumed.resumed_from_epoch == 2
    assert resumed.epoch_losses == first.epoch_losses
    assert resumed.learning_rates == []


def test_resume_continues_lr_schedule(tmp_path) -> None:
    ckpt = tmp_path / "ckpt.pt"
    decay = {
        "batch_size": 2,
        "hidden_dim": 8,
        "learning_rate": 1e-2,
        "lr_decay_factor": 0.5,
        "lr_decay_every": 1,
        "seed": 3,
    }
    train_synthetic(TrainingConfig(n_epochs=2, **decay), _factory, save_to=ckpt)
    resumed = train_synthetic(TrainingConfig(n_epochs=4, **decay), _factory, resume_from=ckpt)
    # only epochs 2 and 3 run after the resume, continuing the decay
    assert resumed.learning_rates == pytest.approx([1e-2 * 0.5**2, 1e-2 * 0.5**3])
