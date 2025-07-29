"""Golden Markdown outputs; drift must be deliberate."""

from __future__ import annotations

from wavrail.ctc._common import Hypothesis
from wavrail.metrics import aggregate_score
from wavrail.report.export import (
    hypotheses_to_markdown,
    hypothesis_record,
    markdown_table,
    score_report_to_markdown,
)


def test_markdown_table_golden() -> None:
    out = markdown_table(["name", "cer"], [["utt|1", 0.5], ["utt2", "1.0"]])
    assert out == ("| name | cer |\n| --- | --- |\n| utt\\|1 | 0.5 |\n| utt2 | 1.0 |\n")


def test_hypotheses_markdown_golden() -> None:
    hyp = Hypothesis(tokens=(1,), text="a", confidence=0.5)
    md = hypotheses_to_markdown([hypothesis_record("u1", hyp)])
    assert md == (
        "# hypotheses\n"
        "\n"
        "| name | text | confidence | tokens |\n"
        "| --- | --- | --- | --- |\n"
        "| u1 | a | 0.500000 | 1 |\n"
    )


def test_score_markdown_golden() -> None:
    corpus = aggregate_score("character")
    corpus.add("u1", "ab", "a")
    md = score_report_to_markdown(corpus.as_report(title="t"))
    assert md == (
        "# t\n"
        "\n"
        "- unit: character\n"
        "- utterances: 1\n"
        "- total errors: 1\n"
        "- total length: 2\n"
        "- error rate: 0.500000\n"
        "\n"
        "| name | errors | length | sub | ins | del | rate |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| u1 | 1 | 2 | 0 | 0 | 1 | 0.5000 |\n"
    )
