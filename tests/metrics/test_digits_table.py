"""Exhaustive digit normalization table in both directions."""

from __future__ import annotations

import pytest

from wavrail.metrics import normalize_digits

CHINESE = "零一二三四五六七八九"


@pytest.mark.parametrize("digit", range(10))
def test_arabic_to_chinese(digit: int) -> None:
    assert normalize_digits(str(digit), to_chinese=True) == CHINESE[digit]


@pytest.mark.parametrize("digit", range(10))
def test_chinese_to_arabic(digit: int) -> None:
    assert normalize_digits(CHINESE[digit]) == str(digit)


@pytest.mark.parametrize("digit", range(10))
def test_roundtrip_both_directions(digit: int) -> None:
    assert normalize_digits(normalize_digits(str(digit), to_chinese=True)) == str(digit)


def test_mixed_text_only_touches_digits() -> None:
    assert normalize_digits("2026年", to_chinese=True) == "二零二六年"
    assert normalize_digits("二零二六年") == "2026年"


def test_non_digit_text_unchanged() -> None:
    assert normalize_digits("abc 你好", to_chinese=True) == "abc 你好"
    assert normalize_digits("abc 你好") == "abc 你好"
