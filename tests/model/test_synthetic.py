"""Tests for the synthetic dataset builder."""

from __future__ import annotations

from wavrail.model import build_synthetic_dataset
from wavrail.vocab import Vocab


def test_synthetic_dataset_is_deterministic() -> None:
    vocab = Vocab.from_labels(["<blank>", "a", "b", "c"])
    a = build_synthetic_dataset(vocab, n_examples=4, duration_s=0.2, seed=7)
    b = build_synthetic_dataset(vocab, n_examples=4, duration_s=0.2, seed=7)
    assert len(a) == 4
    assert [e.text for e in a] == [e.text for e in b]


def test_synthetic_dataset_features_shape() -> None:
    vocab = Vocab.from_labels(["<blank>", "a", "b"])
    examples = build_synthetic_dataset(vocab, n_examples=3, duration_s=0.2)
    for example in examples:
        assert example.features.ndim == 2
        assert example.features.shape[0] > 0
        assert example.features.shape[1] == 40


def test_synthetic_dataset_audio_length() -> None:
    vocab = Vocab.from_labels(["<blank>", "a"])
    examples = build_synthetic_dataset(vocab, n_examples=2, duration_s=0.25, sample_rate=16000)
    for example in examples:
        assert example.audio.shape[0] == 4000


def test_synthetic_dataset_rejects_empty_vocab() -> None:
    vocab = Vocab.from_labels(["<blank>"])
    try:
        build_synthetic_dataset(vocab, n_examples=1)
    except Exception as exc:
        assert "non-blank" in str(exc).lower() or "vocab" in str(exc).lower()
    else:
        raise AssertionError("expected an exception")
