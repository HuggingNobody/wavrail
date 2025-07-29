"""Markdown rendering of forced-alignment records."""

from __future__ import annotations

from wavrail.align import Alignment
from wavrail.report.export import alignment_record, alignments_to_markdown


def _alignment() -> Alignment:
    return Alignment(
        tokens=(1, 2),
        frame_spans=((0, 3), (3, 5)),
        log_probabilities=(-1.5, -0.25),
        confidences=(0.9, 0.8),
        path=(),
    )


def test_alignment_record_carries_schema_fields() -> None:
    record = alignment_record("utt1", _alignment())
    assert record["name"] == "utt1"
    assert record["kind"] == "alignment"
    assert record["frame_spans"] == [[0, 3], [3, 5]]


def test_alignments_markdown_one_row_per_token() -> None:
    md = alignments_to_markdown([alignment_record("utt1", _alignment())], title="align")
    assert md.startswith("# align\n\n")
    assert "| utt1 | 1 | 0-3 | 0.900000 |" in md
    assert "| utt1 | 2 | 3-5 | 0.800000 |" in md


def test_alignments_markdown_handles_empty_list() -> None:
    md = alignments_to_markdown([])
    assert md.endswith("| --- | --- | --- | --- |\n")
