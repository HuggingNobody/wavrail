"""Roundtrip test for vocab serialisation."""

from __future__ import annotations

from wavrail.vocab import Vocab


def test_vocab_serde_roundtrip() -> None:
    vocab = Vocab.from_labels(["<blank>", "a", "b", "c"], name="demo")
    payload = vocab.to_dict()
    restored = Vocab.from_dict(payload)
    assert restored.labels == vocab.labels
    assert restored.blank_index == vocab.blank_index
    assert restored.name == vocab.name


def test_vocab_serde_unicode() -> None:
    vocab = Vocab.from_labels(["<blank>", "\u4f60", "\u597d"], name="cjk")
    payload = vocab.to_dict()
    restored = Vocab.from_dict(payload)
    assert restored.labels == vocab.labels
