"""CLI-written exports must be readable by the library readers."""

from __future__ import annotations

from wavrail.cli.main import EXIT_OK, main
from wavrail.report.export import read_jsonl
from wavrail.report.schema import validate_score_report


def test_score_jsonl_file_validates(tmp_path) -> None:
    target = tmp_path / "scores.jsonl"
    code = main(["score", "abc", "axc", "--format", "jsonl", "--output", str(target)])
    assert code == EXIT_OK
    records = read_jsonl(target)
    assert len(records) == 1
    validate_score_report(records[0])
    assert records[0]["total_errors"] == 1


def test_score_markdown_file_written(tmp_path) -> None:
    target = tmp_path / "report.md"
    code = main(["score", "abc", "abc", "--format", "md", "--output", str(target)])
    assert code == EXIT_OK
    text = target.read_text(encoding="utf-8")
    assert text.startswith("# cli-score")
    assert text.endswith("\n")
    assert "- error rate: 0.000000" in text
