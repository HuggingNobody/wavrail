"""CorpusScore.rate_summary statistics."""

from __future__ import annotations

import pytest

from wavrail.errors import MetricError
from wavrail.metrics import RateSummary, aggregate_score


def test_summary_statistics():
    corpus = aggregate_score()
    corpus.add("u1", "aaaa", "aaaa")  # rate 0.0
    corpus.add("u2", "bbbb", "bbbx")  # rate 0.25
    corpus.add("u3", "cccc", "ccdd")  # rate 0.5
    summary = corpus.rate_summary()
    assert isinstance(summary, RateSummary)
    assert summary.count == 3
    assert summary.min_rate == 0.0
    assert summary.max_rate == pytest.approx(0.5)
    assert summary.mean_rate == pytest.approx(0.25)
    assert summary.median_rate == pytest.approx(0.25)


def test_median_of_even_count_averages_middle():
    corpus = aggregate_score()
    corpus.add("u1", "aa", "aa")  # rate 0.0
    corpus.add("u2", "bb", "bx")  # rate 0.5
    assert corpus.rate_summary().median_rate == pytest.approx(0.25)


def test_empty_reference_utterances_are_skipped():
    corpus = aggregate_score()
    corpus.add("empty", "", "")
    corpus.add("u1", "ab", "ab")
    assert corpus.rate_summary().count == 1


def test_empty_corpus_rejected():
    with pytest.raises(MetricError):
        aggregate_score().rate_summary()
