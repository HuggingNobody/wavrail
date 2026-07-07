"""corpus_to_markdown rendering, escaping, and empty-corpus policy."""

from __future__ import annotations

import pytest

from wavrail.errors import MetricError
from wavrail.metrics import aggregate_score
from wavrail.report import corpus_to_markdown


def _corpus():
    corpus = aggregate_score()
    corpus.add("utt|1", "aaaa", "aaab", duration_seconds=1.5)
    corpus.add("utt2", "bb", "b", noise_level_db=-5.0)
    return corpus


def test_header_carries_summary_statistics():
    md = corpus_to_markdown(_corpus(), title="dev")
    assert md.startswith("# dev\n")
    assert "- unit: character" in md
    assert "- utterances: 2" in md
    assert "- aggregate error rate:" in md
    assert "- median per-utterance rate:" in md


def test_table_rows_and_escaping():
    md = corpus_to_markdown(_corpus())
    assert "| name | errors | length | rate | duration_s | noise_db |" in md
    assert "utt\\|1" in md  # pipes inside cells are escaped
    assert "1.500" in md
    assert "-5.0" in md


def test_empty_corpus_rejected():
    with pytest.raises(MetricError):
        corpus_to_markdown(aggregate_score())
