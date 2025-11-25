"""Vocab.from_text character-level builder."""

from __future__ import annotations

import pytest

from wavrail.errors import VocabularyError
from wavrail.vocab import Vocab


def test_chinese_first_seen_order():
    v = Vocab.from_text("你好世界你好")
    assert v.labels == ("<blank>", "你", "好", "世", "界")
    assert v.token_to_id("世") == 3


def test_english_characters_and_space():
    v = Vocab.from_text("ab ba")
    assert v.labels == ("<blank>", "a", "b", " ")


def test_custom_blank_token():
    v = Vocab.from_text("ab", blank_token="|")
    assert v.labels[0] == "|"
    assert v.is_blank(0)


def test_blank_token_characters_are_ordinary_labels():
    v = Vocab.from_text("<blank>a")
    assert len(v.labels) == len(set(v.labels))
    assert v.token_to_id("<") == 1


def test_rejects_text_without_new_characters():
    with pytest.raises(VocabularyError):
        Vocab.from_text("")
    with pytest.raises(VocabularyError):
        Vocab.from_text("x", blank_token="x")
