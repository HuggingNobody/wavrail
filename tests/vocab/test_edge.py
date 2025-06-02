"""Vocab error-path tests."""

from __future__ import annotations

import pytest

from wavrail.errors import VocabularyError
from wavrail.vocab import Vocab


def test_rejects_empty_labels() -> None:
    with pytest.raises(VocabularyError):
        Vocab.from_labels([])


def test_rejects_duplicate_labels() -> None:
    with pytest.raises(VocabularyError):
        Vocab.from_labels(["<blank>", "a", "a"])


def test_rejects_blank_index_out_of_range() -> None:
    with pytest.raises(VocabularyError):
        Vocab(labels=("<blank>", "a"), blank_index=5)


def test_unknown_token_raises() -> None:
    vocab = Vocab.from_labels(["<blank>", "a", "b"])
    with pytest.raises(VocabularyError):
        vocab.token_to_id("z")


def test_index_out_of_range() -> None:
    vocab = Vocab.from_labels(["<blank>", "a", "b"])
    with pytest.raises(VocabularyError):
        vocab.id_to_token(99)


def test_is_blank_helper() -> None:
    vocab = Vocab.from_labels(["x", "a", "b"], blank_index=1)
    assert vocab.is_blank(1)
    assert not vocab.is_blank(0)


def test_size_and_len() -> None:
    vocab = Vocab.from_labels(["<blank>", "a", "b", "c"])
    assert vocab.size == 4
    assert len(vocab) == 4
