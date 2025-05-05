"""Tests for corpus-level aggregation."""

from __future__ import annotations

from wavrail.metrics import aggregate_score


def test_aggregate_empty() -> None:
    score = aggregate_score()
    assert score.total_errors == 0
    assert score.error_rate == 0.0


def test_aggregate_total_errors_and_rate() -> None:
    score = aggregate_score(unit="character")
    score.add("u1", "hello world", "hello world")
    score.add("u2", "abc", "axc")
    assert score.total_length == 11 + 3
    assert score.total_errors == 1
    assert abs(score.error_rate - 1 / 14) < 1e-9


def test_slice_by_duration() -> None:
    score = aggregate_score()
    score.add("a", "abc", "abc", duration_seconds=1.0)
    score.add("b", "abc", "xyz", duration_seconds=5.0)
    slices = score.slice_by_duration(2.0)
    assert slices["short"] == 0.0
    assert abs(slices["long"] - 1.0) < 1e-9


def test_slice_by_noise() -> None:
    score = aggregate_score()
    score.add("a", "abc", "abc", noise_level_db=-20.0)
    score.add("b", "abc", "xyz", noise_level_db=20.0)
    slices = score.slice_by_noise(0.0)
    assert slices["clean"] == 0.0
    assert abs(slices["noisy"] - 1.0) < 1e-9


def test_report_contains_unit_and_counts() -> None:
    score = aggregate_score(unit="word")
    score.add("u1", "a b c", "a b c")
    report = score.as_report()
    assert report["unit"] == "word"
    assert report["total_errors"] == 0
    assert report["total_length"] == 3
