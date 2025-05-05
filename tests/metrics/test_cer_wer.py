"""Tests for CER/WER and edit alignment."""

from __future__ import annotations

from wavrail.metrics import (
    character_error_rate,
    edit_alignment,
    word_error_rate,
)


def test_cer_identical_is_zero() -> None:
    assert character_error_rate("hello world", "hello world") == 0.0


def test_cer_single_substitution() -> None:
    assert abs(character_error_rate("abc", "axc") - 1 / 3) < 1e-9


def test_cer_empty_reference_with_hypothesis() -> None:
    assert character_error_rate("", "abc") == float("inf")


def test_cer_empty_both() -> None:
    assert character_error_rate("", "") == 0.0


def test_wer_word_substitution() -> None:
    assert abs(word_error_rate("a b c", "a d c") - 1 / 3) < 1e-9


def test_wer_insertion_deletion() -> None:
    assert abs(word_error_rate("a b", "a b c") - 1 / 2) < 1e-9
    assert abs(word_error_rate("a b c", "a b") - 1 / 3) < 1e-9


def test_edit_alignment_counts() -> None:
    ops = edit_alignment(["a", "b", "c"], ["a", "x", "c"])
    assert ops.substitutions == 1
    assert ops.insertions == 0
    assert ops.deletions == 0


def test_edit_alignment_insertion_only() -> None:
    ops = edit_alignment(["a"], ["a", "b"])
    assert ops.insertions == 1
    assert ops.errors == 1
