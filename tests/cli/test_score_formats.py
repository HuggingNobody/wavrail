"""score --format json|jsonl|md output paths."""

from __future__ import annotations

import json

from wavrail.cli.main import EXIT_OK, main
from wavrail.report.export import read_jsonl
from wavrail.report.schema import score_report_from_json


def test_score_json_format(capsys) -> None:
    code = main(["score", "abc", "axc", "--format", "json"])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "1"
    assert payload["unit"] == "character"
    assert payload["total_errors"] == 1
    assert len(payload["utterances"]) == 1


def test_score_jsonl_format_single_line(capsys) -> None:
    code = main(["score", "abc", "abc", "--format", "jsonl"])
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert out.count("\n") == 1
    assert json.loads(out)["error_rate"] == 0.0


def test_score_markdown_format(capsys) -> None:
    code = main(["score", "你好", "你", "--format", "md"])
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert out.startswith("# cli-score")
    assert "| cli | 1 | 2 | 0 | 0 | 1 | 0.5000 |" in out


def test_score_text_format_unchanged(capsys) -> None:
    code = main(["score", "abc", "axc"])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"unit", "rate"}


def test_score_output_to_file(tmp_path) -> None:
    target = tmp_path / "nested" / "report.jsonl"
    code = main(["score", "abc", "axc", "--format", "jsonl", "--output", str(target)])
    assert code == EXIT_OK
    records = read_jsonl(target)
    assert len(records) == 1
    assert records[0]["total_errors"] == 1


def test_cli_json_output_passes_strict_validation(capsys) -> None:
    main(["score", "hello world", "hello", "--unit", "word", "--format", "json"])
    text = capsys.readouterr().out
    payload = score_report_from_json(text)
    assert payload["unit"] == "word"
    assert payload["total_errors"] == 1
