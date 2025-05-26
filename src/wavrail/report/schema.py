"""JSON-friendly representations of wavrail public outputs."""

from __future__ import annotations

import json
from typing import Any

from wavrail.align import Alignment
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


def score_report_to_json(score: CorpusScore, *, title: str = "score") -> str:
    payload = score.as_report(title=title)
    payload["schema_version"] = SCHEMA_VERSION
    return json.dumps(payload, ensure_ascii=False, indent=2)


def score_report_from_json(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    _check_version(payload)
    return payload


def _check_version(payload: dict[str, Any]) -> None:
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SchemaError(f"incompatible schema_version={version!r}; expected {SCHEMA_VERSION!r}")


__all__ = [
    "SCHEMA_VERSION",
    "align_from_dict",
    "align_to_dict",
    "hypothesis_from_dict",
    "hypothesis_to_dict",
    "score_report_from_json",
    "score_report_to_json",
]
