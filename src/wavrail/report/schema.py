"""JSON-friendly representations of wavrail public outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from wavrail.align import Alignment, summarize_alignment
from wavrail.ctc._common import Hypothesis
from wavrail.errors import SchemaError
from wavrail.metrics import CorpusScore

SCHEMA_VERSION = "1"


def _with_schema(payload: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, **payload}


def hypothesis_to_dict(hyp: Hypothesis) -> dict[str, Any]:
    return _with_schema(
        {
            "kind": "hypothesis",
            "tokens": list(hyp.tokens),
            "text": hyp.text,
            "confidence": hyp.confidence,
            "frame_indices": list(hyp.frame_indices),
        }
    )


def hypothesis_from_dict(payload: dict[str, Any]) -> Hypothesis:
    _check_version(payload)
    if payload.get("kind") != "hypothesis":
        raise SchemaError(f"expected kind=hypothesis, got {payload.get('kind')!r}")
    return Hypothesis(
        tokens=tuple(int(t) for t in payload.get("tokens", [])),
        text=str(payload.get("text", "")),
        confidence=float(payload.get("confidence", 0.0)),
        frame_indices=tuple(int(i) for i in payload.get("frame_indices", [])),
    )


def align_to_dict(alignment: Alignment) -> dict[str, Any]:
    return _with_schema(
        {
            "kind": "alignment",
            "tokens": list(alignment.tokens),
            "frame_spans": [list(span) for span in alignment.frame_spans],
            "log_probabilities": list(alignment.log_probabilities),
            "confidences": list(alignment.confidences),
            "path": list(alignment.path),
        }
    )


def align_from_dict(payload: dict[str, Any]) -> Alignment:
    _check_version(payload)
    if payload.get("kind") != "alignment":
        raise SchemaError(f"expected kind=alignment, got {payload.get('kind')!r}")
    spans = payload.get("frame_spans", [])
    return Alignment(
        tokens=tuple(int(t) for t in payload.get("tokens", [])),
        frame_spans=tuple((int(a), int(b)) for a, b in spans),
        log_probabilities=tuple(float(v) for v in payload.get("log_probabilities", [])),
        confidences=tuple(float(v) for v in payload.get("confidences", [])),
        path=tuple(int(p) for p in payload.get("path", [])),
    )


def score_report_to_dict(score: CorpusScore, *, title: str = "score") -> dict[str, Any]:
    """Score report as a plain dict including the schema version."""
    payload = score.as_report(title=title)
    payload["schema_version"] = SCHEMA_VERSION
    return payload


def score_report_to_json(score: CorpusScore, *, title: str = "score") -> str:
    return json.dumps(score_report_to_dict(score, title=title), ensure_ascii=False, indent=2)


def score_report_from_json(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    _check_version(payload)
    validate_score_report(payload)
    return payload


_UTTERANCE_KEYS = ("name", "errors", "length", "insertions", "deletions", "substitutions")


def validate_score_report(payload: Mapping[str, Any]) -> None:
    """Strict structural validation of a score-report payload.

    Checks the required keys, the unit vocabulary, per-utterance
    field presence, the ``errors = substitutions + insertions +
    deletions`` identity, and that the aggregate totals and error
    rate are consistent with the utterance rows. Any violation
    raises :class:`SchemaError` so corrupt or hand-edited reports
    never silently propagate.
    """
    for key in ("title", "unit", "utterances", "total_errors", "total_length", "error_rate"):
        if key not in payload:
            raise SchemaError(f"score report is missing key {key!r}")
    if payload["unit"] not in {"character", "word"}:
        raise SchemaError(f"unknown unit {payload['unit']!r}")
    utterances = payload["utterances"]
    if not isinstance(utterances, list):
        raise SchemaError("utterances must be a list")
    total_errors = 0
    total_length = 0
    for index, utterance in enumerate(utterances):
        if not isinstance(utterance, dict):
            raise SchemaError(f"utterance {index} must be an object")
        for key in _UTTERANCE_KEYS:
            if key not in utterance:
                raise SchemaError(f"utterance {index} is missing key {key!r}")
        errors = int(utterance["errors"])
        parts = (
            int(utterance["insertions"])
            + int(utterance["deletions"])
            + int(utterance["substitutions"])
        )
        if errors != parts:
            raise SchemaError(f"utterance {index}: errors ({errors}) != sub+ins+del ({parts})")
        total_errors += errors
        total_length += int(utterance["length"])
    if int(payload["total_errors"]) != total_errors:
        raise SchemaError("total_errors does not match the utterance sum")
    if int(payload["total_length"]) != total_length:
        raise SchemaError("total_length does not match the utterance sum")
    expected_rate = (total_errors / total_length) if total_length else 0.0
    if abs(float(payload["error_rate"]) - expected_rate) > 1e-9:
        raise SchemaError("error_rate does not match the totals")


def _check_version(payload: dict[str, Any]) -> None:
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SchemaError(f"incompatible schema_version={version!r}; expected {SCHEMA_VERSION!r}")


def alignment_summary_to_dict(alignment: Alignment, frame_shift_s: float) -> dict[str, Any]:
    """Versioned seconds-domain summary payload for an alignment.

    Wraps :func:`wavrail.align.summarize_alignment` in the schema
    envelope and records the frame shift used for the conversion, so
    readers can reproduce every seconds-based number.
    """
    summary = summarize_alignment(alignment, frame_shift_s)
    return _with_schema({"kind": "alignment_summary", "frame_shift_s": frame_shift_s, **summary})


__all__ = [
    "SCHEMA_VERSION",
    "align_from_dict",
    "align_to_dict",
    "alignment_summary_to_dict",
    "hypothesis_from_dict",
    "hypothesis_to_dict",
    "score_report_from_json",
    "score_report_to_dict",
    "score_report_to_json",
    "validate_score_report",
]
