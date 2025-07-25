"""Hypothesis records and their Markdown rendering."""

from __future__ import annotations

from wavrail.ctc._common import Hypothesis
from wavrail.report.export import hypotheses_to_markdown, hypothesis_record


def test_hypothesis_record_carries_schema_fields() -> None:
    hyp = Hypothesis(tokens=(1, 2), text="ab", confidence=0.5, frame_indices=(0, 3))
    record = hypothesis_record("utt1", hyp)
    assert record["name"] == "utt1"
    assert record["kind"] == "hypothesis"
    assert record["schema_version"] == "1"
    assert record["tokens"] == [1, 2]
    assert record["frame_indices"] == [0, 3]


def test_hypotheses_markdown_lists_every_record() -> None:
    hyp = Hypothesis(tokens=(1,), text="a|b", confidence=0.25)
    md = hypotheses_to_markdown([hypothesis_record("u1", hyp)], title="decode")
    assert md.startswith("# decode\n\n")
    assert "| u1 | a\\|b | 0.250000 | 1 |" in md


def test_hypotheses_markdown_handles_empty_list() -> None:
    md = hypotheses_to_markdown([])
    assert (
        md == "# hypotheses\n\n| name | text | confidence | tokens |\n| --- | --- | --- | --- |\n"
    )
