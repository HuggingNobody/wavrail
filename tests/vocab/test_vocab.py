"""Tests for the :class:`wavrail.vocab.Vocab` class."""

from __future__ import annotations

import pytest

from wavrail.errors import VocabularyError
from wavrail.vocab import Vocab


def test_basic_mapping() -> None:
    v = Vocab.from_labels(["<blank>", "a", "b", "c"])
    assert v.size == 4
    assert v.token_to_id("a") == 1
    assert v.id_to_token(2) == "b"
    assert v.is_blank(0)
    assert not v.is_blank(1)


def test_unicode_chinese_labels() -> None:
    v = Vocab.from_labels(["<blank>", "你", "好", "世", "界"])
    assert v.token_to_id("好") == 2
    assert v.id_to_token(3) == "世"


def test_rejects_duplicate_labels() -> None:
    with pytest.raises(VocabularyError):
        Vocab.from_labels(["a", "a"])


def test_rejects_empty_labels() -> None:
    with pytest.raises(VocabularyError):
        Vocab.from_labels([])


def test_rejects_invalid_blank_index() -> None:
    with pytest.raises(VocabularyError):
        Vocab.from_labels(["a", "b"], blank_index=2)
    with pytest.raises(VocabularyError):
        Vocab.from_labels(["a", "b"], blank_index=-1)


def test_unknown_token_raises() -> None:
    v = Vocab.from_labels(["<blank>", "a"])
    with pytest.raises(VocabularyError):
        v.token_to_id("z")


def test_index_out_of_range_raises() -> None:
    v = Vocab.from_labels(["<blank>", "a"])
    with pytest.raises(VocabularyError):
        v.id_to_token(5)
