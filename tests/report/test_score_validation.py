"""Strict validation of score-report payloads."""

from __future__ import annotations

import json

import pytest

from wavrail.errors import SchemaError
from wavrail.metrics import aggregate_score
from wavrail.report.schema import (
    score_report_from_json,
    score_report_to_dict,
    validate_score_report,
)


def _payload() -> dict:
    corpus = aggregate_score("character")
    corpus.add("u1", "abc", "axc")
    corpus.add("u2", "你好", "你")
    return score_report_to_dict(corpus, title="eval")


def test_valid_payload_passes() -> None:
    validate_score_report(_payload())


@pytest.mark.parametrize(
    "key", ["title", "unit", "utterances", "total_errors", "total_length", "error_rate"]
)
def test_missing_top_level_key_rejected(key: str) -> None:
    payload = _payload()
    del payload[key]
    with pytest.raises(SchemaError, match="missing key"):
        validate_score_report(payload)


def test_unknown_unit_rejected() -> None:
    payload = _payload()
    payload["unit"] = "phone"
    with pytest.raises(SchemaError, match="unknown unit"):
        validate_score_report(payload)


def test_errors_identity_enforced() -> None:
    payload = _payload()
    payload["utterances"][0]["substitutions"] = 5
    with pytest.raises(SchemaError, match="sub\\+ins\\+del"):
        validate_score_report(payload)


def test_total_errors_mismatch_rejected() -> None:
    payload = _payload()
    payload["total_errors"] = 99
    with pytest.raises(SchemaError, match="total_errors"):
        validate_score_report(payload)


def test_error_rate_mismatch_rejected() -> None:
    payload = _payload()
    payload["error_rate"] = 0.9
    with pytest.raises(SchemaError, match="error_rate"):
        validate_score_report(payload)


def test_utterance_missing_key_rejected() -> None:
    payload = _payload()
    del payload["utterances"][0]["name"]
    with pytest.raises(SchemaError, match="utterance 0"):
        validate_score_report(payload)


def test_non_list_utterances_rejected() -> None:
    payload = _payload()
    payload["utterances"] = {}
    with pytest.raises(SchemaError, match="must be a list"):
        validate_score_report(payload)


def test_from_json_validates_body() -> None:
    payload = _payload()
    payload["total_length"] = 77
    with pytest.raises(SchemaError, match="total_length"):
        score_report_from_json(json.dumps(payload))


def test_from_json_accepts_valid_report() -> None:
    text = json.dumps(_payload(), ensure_ascii=False)
    assert score_report_from_json(text)["title"] == "eval"
