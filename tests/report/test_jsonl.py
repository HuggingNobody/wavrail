"""JSONL serialisation: escaping, append mode, and error handling."""

from __future__ import annotations

import pytest

from wavrail.errors import SchemaError
from wavrail.report.export import read_jsonl, to_jsonl, write_jsonl


def test_roundtrip_preserves_unicode_and_newlines(tmp_path) -> None:
    records = [
        {"name": "utt1", "text": "你好, world\nsecond line"},
        {"name": "utt2", "text": 'quote " and backslash \\'},
    ]
    path = tmp_path / "records.jsonl"
    assert write_jsonl(records, path) == 2
    assert read_jsonl(path) == records


def test_lines_never_contain_raw_newlines(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    write_jsonl([{"text": "a\nb\rc"}], path)
    raw = path.read_text(encoding="utf-8")
    assert raw.count("\n") == 1  # only the record terminator
    assert "\r" not in raw


def test_append_mode_extends_file(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    write_jsonl([{"i": 1}], path)
    write_jsonl([{"i": 2}], path, append=True)
    assert read_jsonl(path) == [{"i": 1}, {"i": 2}]


def test_read_skips_blank_lines(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"i": 1}\n\n{"i": 2}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"i": 1}, {"i": 2}]


def test_read_rejects_invalid_json_with_line_number(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"i": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(SchemaError, match="line 2"):
        read_jsonl(path)


def test_read_rejects_non_object_lines(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="not a JSON object"):
        read_jsonl(path)


def test_read_missing_file_raises(tmp_path) -> None:
    with pytest.raises(SchemaError, match="not found"):
        read_jsonl(tmp_path / "nope.jsonl")


def test_to_jsonl_empty_is_empty_string() -> None:
    assert to_jsonl([]) == ""
