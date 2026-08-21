"""Golden snapshot of corpus_to_markdown; drift must be deliberate."""

from __future__ import annotations

from wavrail.metrics import aggregate_score
from wavrail.report import corpus_to_markdown

GOLDEN = (
    "# dev\n"
    "\n"
    "- unit: character\n"
    "- utterances: 2\n"
    "- aggregate error rate: 0.333333\n"
    "- mean per-utterance rate: 0.375000\n"
    "- median per-utterance rate: 0.375000\n"
    "- min/max rate: 0.250000 / 0.500000\n"
    "\n"
    "| name | errors | length | rate | duration_s | noise_db |\n"
    "| --- | --- | --- | --- | --- | --- |\n"
    "| u1 | 1 | 4 | 0.250000 | 1.500 |  |\n"
    "| u2 | 1 | 2 | 0.500000 |  | -5.0 |\n"
)


def test_golden_snapshot():
    corpus = aggregate_score()
    corpus.add("u1", "aaaa", "aaab", duration_seconds=1.5)
    corpus.add("u2", "bb", "b", noise_level_db=-5.0)
    assert corpus_to_markdown(corpus, title="dev") == GOLDEN
