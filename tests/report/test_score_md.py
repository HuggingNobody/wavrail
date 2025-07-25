"""Markdown rendering of score reports."""

from __future__ import annotations

from wavrail.metrics import aggregate_score
from wavrail.report.export import score_report_to_markdown


def _payload() -> dict:
    corpus = aggregate_score("character")
    corpus.add("u1", "abc", "axc")
    corpus.add("u2", "你好", "你")
    return corpus.as_report(title="eval")


def test_score_markdown_summarises_corpus() -> None:
    md = score_report_to_markdown(_payload())
    assert "# eval" in md
    assert "- unit: character" in md
    assert "- utterances: 2" in md
    assert "- total errors: 2" in md
    assert "- total length: 5" in md
    assert "- error rate: 0.400000" in md


def test_score_markdown_lists_per_utterance_rows() -> None:
    md = score_report_to_markdown(_payload())
    assert "| u1 | 1 | 3 | 1 | 0 | 0 | 0.3333 |" in md
    assert "| u2 | 1 | 2 | 0 | 0 | 1 | 0.5000 |" in md


def test_score_markdown_empty_corpus() -> None:
    md = score_report_to_markdown(aggregate_score("word").as_report())
    assert "- utterances: 0" in md
    assert "- error rate: 0.000000" in md
