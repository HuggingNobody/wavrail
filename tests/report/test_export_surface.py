"""The report package surface is pinned; drift must be deliberate."""

from __future__ import annotations

import wavrail.report as report

EXPECTED = {
    "SCHEMA_VERSION",
    "SchemaError",
    "align_from_dict",
    "align_to_dict",
    "alignment_record",
    "alignments_to_markdown",
    "hypotheses_to_markdown",
    "hypothesis_from_dict",
    "hypothesis_record",
    "hypothesis_to_dict",
    "markdown_table",
    "read_jsonl",
    "score_report_from_json",
    "score_report_to_dict",
    "score_report_to_json",
    "score_report_to_markdown",
    "to_jsonl",
    "validate_score_report",
    "write_jsonl",
}


def test_all_is_pinned() -> None:
    assert set(report.__all__) == EXPECTED


def test_every_export_is_importable() -> None:
    for name in report.__all__:
        assert hasattr(report, name), name
