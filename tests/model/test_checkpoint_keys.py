"""Golden key sets for saved checkpoints; drift must be deliberate."""

from __future__ import annotations

import pytest

from wavrail.model import TrainingConfig, build_synthetic_dataset, train_synthetic
from wavrail.vocab import Vocab

pytestmark = pytest.mark.model


def test_saved_checkpoint_keys_are_stable(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    vocab = Vocab.from_labels(["<blank>", "a", "b", "c"])

    def factory(seed: int) -> tuple[list, list]:
        examples = build_synthetic_dataset(vocab, n_examples=4, duration_s=0.2, seed=seed)
        return examples[:3], examples[3:]

    ckpt = tmp_path / "ckpt.pt"
    train_synthetic(TrainingConfig(n_epochs=1, hidden_dim=8), factory, save_to=ckpt)
    payload = torch.load(ckpt, weights_only=False)
    assert set(payload) == {
        "schema_version",
        "epoch",
        "config",
        "model_state",
        "optimizer_state",
        "epoch_losses",
    }
    assert set(payload["config"]) == {
        "n_epochs",
        "batch_size",
        "learning_rate",
        "grad_clip",
        "hidden_dim",
        "seed",
        "lr_decay_factor",
        "lr_decay_every",
    }
    assert set(payload["model_state"]) == {
        "gru",
        "head",
        "input_dim",
        "hidden_dim",
        "vocab_size",
    }
    assert payload["epoch"] == 1
    assert len(payload["epoch_losses"]) == 1
