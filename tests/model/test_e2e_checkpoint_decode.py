"""Train, checkpoint, reload, and decode end to end (torch extra)."""

from __future__ import annotations

import pytest

from wavrail.model import (
    TrainingConfig,
    build_synthetic_dataset,
    decode_example,
    load_checkpoint,
    train_synthetic,
)
from wavrail.model.network import AcousticModel
from wavrail.vocab import Vocab

pytestmark = pytest.mark.model


def test_trained_checkpoint_decodes_examples(tmp_path) -> None:
    vocab = Vocab.from_labels(["<blank>", "a", "b", "c"])

    def factory(seed: int) -> tuple[list, list]:
        examples = build_synthetic_dataset(vocab, n_examples=4, duration_s=0.2, seed=seed)
        return examples[:3], examples[3:]

    ckpt = tmp_path / "e2e.pt"
    config = TrainingConfig(n_epochs=2, batch_size=2, hidden_dim=8)
    report = train_synthetic(config, factory, vocab=vocab, save_to=ckpt)
    assert report.loss_decreased()

    checkpoint = load_checkpoint(ckpt)
    model = AcousticModel(
        input_dim=checkpoint.model_state["input_dim"],
        hidden_dim=checkpoint.model_state["hidden_dim"],
        vocab_size=checkpoint.model_state["vocab_size"],
    )
    model.load_state_dict(checkpoint.model_state)

    dev_examples = factory(config.seed)[1]
    hyp = decode_example(model, dev_examples[0], vocab)
    assert isinstance(hyp.text, str)
    hyp_beam = decode_example(model, dev_examples[0], vocab, beam=2)
    assert 0.0 <= hyp_beam.confidence <= 1.0
