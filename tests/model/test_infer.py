"""model_posteriors contract and decode_example wiring (torch extra)."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.ctc import Hypothesis
from wavrail.errors import ModelError
from wavrail.model import build_synthetic_dataset, decode_example, model_posteriors
from wavrail.model.network import AcousticModel
from wavrail.vocab import Vocab

pytestmark = pytest.mark.model

VOCAB = Vocab.from_labels(["<blank>", "a", "b", "c"])


def _model(vocab_size: int = 4) -> AcousticModel:
    return AcousticModel(input_dim=40, hidden_dim=8, vocab_size=vocab_size)


def test_posteriors_are_probabilities():
    features = np.random.default_rng(0).standard_normal((23, 40)).astype(np.float32)
    probs = model_posteriors(_model(), features)
    assert probs.shape == (23, 4)
    assert probs.dtype == np.float32
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(probs >= 0.0)


def test_rejects_bad_features():
    with pytest.raises(ModelError):
        model_posteriors(_model(), np.zeros(40, dtype=np.float32))
    with pytest.raises(ModelError):
        model_posteriors(_model(), np.zeros((0, 40), dtype=np.float32))


def test_decode_example_greedy_returns_hypothesis():
    examples = build_synthetic_dataset(VOCAB, n_examples=1, duration_s=0.2, seed=0)
    hyp = decode_example(_model(), examples[0], VOCAB)
    assert isinstance(hyp, Hypothesis)
    assert all(0 <= token < VOCAB.size for token in hyp.tokens)


def test_decode_example_beam_path():
    examples = build_synthetic_dataset(VOCAB, n_examples=1, duration_s=0.2, seed=1)
    hyp = decode_example(_model(), examples[0], VOCAB, beam=2)
    assert isinstance(hyp, Hypothesis)
    assert 0.0 <= hyp.confidence <= 1.0


def test_vocab_size_mismatch_rejected():
    examples = build_synthetic_dataset(VOCAB, n_examples=1, duration_s=0.2, seed=0)
    with pytest.raises(ModelError):
        decode_example(_model(vocab_size=5), examples[0], VOCAB)
