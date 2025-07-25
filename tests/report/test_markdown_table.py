"""Markdown table rendering and cell escaping."""

from __future__ import annotations

import pytest

from wavrail.errors import SchemaError
from wavrail.report.export import markdown_table


def test_basic_table_shape() -> None:
    out = markdown_table(["name", "value"], [["a", 1], ["b", 2]])
    assert out == "| name | value |\n| --- | --- |\n| a | 1 |\n| b | 2 |\n"


def test_cells_escape_pipes_and_backslashes() -> None:
    out = markdown_table(["t"], [["x|y\\z"]])
    assert out.splitlines()[2] == "| x\\|y\\\\z |"


def test_newlines_flatten_to_spaces() -> None:
    out = markdown_table(["t"], [["line1\nline2"]])
    assert out.splitlines()[2] == "| line1 line2 |"


def test_headers_only_table() -> None:
    out = markdown_table(["a", "b"], [])
    assert out == "| a | b |\n| --- | --- |\n"


def test_rejects_ragged_rows() -> None:
    with pytest.raises(SchemaError, match="expected 2"):
        markdown_table(["a", "b"], [[1]])


def test_rejects_empty_headers() -> None:
    with pytest.raises(SchemaError, match="at least one column"):
        markdown_table([], [])
