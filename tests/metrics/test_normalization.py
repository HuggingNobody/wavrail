"""Tests for text normalization helpers."""

from __future__ import annotations

from wavrail.metrics import (
    TextNormalizer,
    fullwidth_to_halfwidth,
    halfwidth_to_fullwidth,
    normalize_digits,
)


def test_fullwidth_to_halfwidth_digit() -> None:
    assert fullwidth_to_halfwidth("1") == "1"
    assert fullwidth_to_halfwidth("\uff21\uff22\uff23") == "ABC"
    assert fullwidth_to_halfwidth("\u3000") == " "


def test_halfwidth_to_fullwidth_roundtrip() -> None:
    text = "abc 123"
    assert fullwidth_to_halfwidth(halfwidth_to_fullwidth(text)) == text


def test_normalize_digits_arabic_to_chinese() -> None:
    assert normalize_digits("0123456789", to_chinese=True) == "零一二三四五六七八九"


def test_normalize_digits_chinese_to_arabic() -> None:
    assert normalize_digits("零一二三四五六七八九") == "0123456789"


def test_text_normalizer_strips_punctuation() -> None:
    norm = TextNormalizer()
    assert norm("Hello, world!") == "hello world"


def test_text_normalizer_case_fold_unicode() -> None:
    norm = TextNormalizer(case_fold=True, strip_punctuation=False)
    assert norm("HEISS") == "heiss"


def test_text_normalizer_fullwidth_and_digits() -> None:
    norm = TextNormalizer(digits_to_chinese=True, strip_punctuation=False)
    assert norm("\uff21\uff22\uff23 1") == "abc 一"
