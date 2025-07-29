"""JSONL and Markdown export for wavrail report payloads.

JSONL (one JSON object per line) is the interchange format for
decoded hypotheses and score reports; Markdown renders the same
payloads as human-readable tables. Both serialisations work fully
offline. Escaping is delegated to ``json.dumps`` for JSONL, and
handled by an explicit cell escaper for Markdown, where pipes,
backslashes, and newlines are the hazardous characters.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from wavrail.align import Alignment
from wavrail.ctc._common import Hypothesis
from wavrail.errors import SchemaError
from wavrail.report.schema import align_to_dict, hypothesis_to_dict


def _dumps(record: Mapping[str, Any]) -> str:
    return json.dumps(dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def to_jsonl(records: Iterable[Mapping[str, Any]]) -> str:
    """Serialise ``records`` to JSONL text (one compact object per line).

    Keys are sorted and separators are minimal so equal input
    records produce byte-identical output, which the golden tests
    rely on. Non-ASCII text stays UTF-8 instead of ``\\u`` escapes
    so files remain readable in any editor.
    """
    lines = [_dumps(record) for record in records]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def write_jsonl(
    records: Iterable[Mapping[str, Any]],
    path: str | Path,
    *,
    append: bool = False,
) -> int:
    """Write ``records`` to ``path`` as JSONL; return the record count.

    ``append=True`` adds to an existing file instead of replacing
    it, so a long evaluation can persist one record per utterance
    as results arrive.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("a" if append else "w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(_dumps(record))
            fh.write("\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Parse a JSONL file into dicts; blank lines are skipped.

    Raises :class:`SchemaError` with the offending line number
    when a line is not valid JSON or is not a JSON object.
    """
    target = Path(path)
    if not target.is_file():
        raise SchemaError(f"jsonl file not found: {target}")
    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"invalid JSON on line {lineno}: {exc}") from exc
            if not isinstance(payload, dict):
                raise SchemaError(f"line {lineno} is not a JSON object")
            records.append(payload)
    return records


def _escape_cell(value: object) -> str:
    text = str(value)
    text = text.replace("\\", "\\\\").replace("|", "\\|")
    return " ".join(text.splitlines())


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """Render a GitHub-flavoured Markdown table with escaped cells.

    Backslashes and pipes are escaped, newlines inside cells are
    flattened to spaces, and every row must match the header width
    so ragged data fails loudly instead of rendering broken tables.
    """
    if not headers:
        raise SchemaError("markdown table needs at least one column")
    width = len(headers)
    lines = ["| " + " | ".join(_escape_cell(h) for h in headers) + " |"]
    lines.append("|" + "|".join(" --- " for _ in headers) + "|")
    for row in rows:
        if len(row) != width:
            raise SchemaError(f"row has {len(row)} cells, expected {width}")
        lines.append("| " + " | ".join(_escape_cell(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"


def hypothesis_record(name: str, hypothesis: Hypothesis) -> dict[str, Any]:
    """Flat JSONL-ready record: the hypothesis schema plus a ``name``."""
    return {"name": name, **hypothesis_to_dict(hypothesis)}


def hypotheses_to_markdown(
    records: Iterable[Mapping[str, Any]], *, title: str = "hypotheses"
) -> str:
    """Render hypothesis records as a Markdown report with one row each."""
    rows = [
        [
            record.get("name", ""),
            record.get("text", ""),
            f"{float(record.get('confidence', 0.0)):.6f}",
            " ".join(str(token) for token in record.get("tokens", [])),
        ]
        for record in records
    ]
    table = markdown_table(["name", "text", "confidence", "tokens"], rows)
    return f"# {title}\n\n{table}"


def score_report_to_markdown(payload: Mapping[str, Any]) -> str:
    """Render a score-report payload (``score_report_to_dict``) as Markdown.

    The summary block mirrors the aggregate fields of the payload;
    the table lists per-utterance counts so regressions can be
    traced to individual utterances.
    """
    utterances = list(payload.get("utterances", []))
    lines = [
        f"# {payload.get('title', 'score')}",
        "",
        f"- unit: {payload.get('unit', 'character')}",
        f"- utterances: {len(utterances)}",
        f"- total errors: {payload.get('total_errors', 0)}",
        f"- total length: {payload.get('total_length', 0)}",
        f"- error rate: {float(payload.get('error_rate', 0.0)):.6f}",
        "",
    ]
    rows = []
    for utterance in utterances:
        length = int(utterance.get("length", 0))
        errors = int(utterance.get("errors", 0))
        rate = f"{errors / length:.4f}" if length else "-"
        rows.append(
            [
                utterance.get("name", ""),
                errors,
                length,
                int(utterance.get("substitutions", 0)),
                int(utterance.get("insertions", 0)),
                int(utterance.get("deletions", 0)),
                rate,
            ]
        )
    table = markdown_table(["name", "errors", "length", "sub", "ins", "del", "rate"], rows)
    return "\n".join(lines) + "\n" + table


def alignment_record(name: str, alignment: Alignment) -> dict[str, Any]:
    """Flat JSONL-ready record: the alignment schema plus a ``name``."""
    return {"name": name, **align_to_dict(alignment)}


def alignments_to_markdown(
    records: Iterable[Mapping[str, Any]], *, title: str = "alignment"
) -> str:
    """Render alignment records with one Markdown row per token."""
    rows = []
    for record in records:
        name = record.get("name", "")
        spans = list(record.get("frame_spans", []))
        confidences = list(record.get("confidences", []))
        for index, token in enumerate(record.get("tokens", [])):
            start, end = spans[index] if index < len(spans) else (0, 0)
            confidence = confidences[index] if index < len(confidences) else 0.0
            rows.append([name, token, f"{start}-{end}", f"{float(confidence):.6f}"])
    table = markdown_table(["name", "token", "frames", "confidence"], rows)
    return f"# {title}\n\n{table}"


__all__ = [
    "alignment_record",
    "alignments_to_markdown",
    "hypotheses_to_markdown",
    "hypothesis_record",
    "markdown_table",
    "read_jsonl",
    "score_report_to_markdown",
    "to_jsonl",
    "write_jsonl",
]
