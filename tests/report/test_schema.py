"""Tests for the schema-versioned serialisation helpers."""

from __future__ import annotations

import json

import pytest

from wavrail.align import align_ctc
from wavrail.ctc import greedy_decode
from wavrail.errors import SchemaError
from wavrail.metrics import aggregate_score
from wavrail.report import (
    SCHEMA_VERSION,
    align_from_dict,
    align_to_dict,
    hypothesis_from_dict,
    hypothesis_to_dict,
    score_report_from_json,
    score_report_to_json,
)
from wavrail.vocab import Vocab


def _vocab() -> Vocab:
    return Vocab.from_labels(["<blank>", "a", "b", "c"])


def test_hypothesis_roundtrip() -> None:
    vocab = _vocab()
    posteriors = __posteriors()
    hyp = greedy_decode(posteriors, vocab)
    restored = hypothesis_from_dict(hypothesis_to_dict(hyp))
    assert restored.tokens == hyp.tokens
    assert restored.text == hyp.text


def test_alignment_roundtrip() -> None:
    vocab = _vocab()
    posteriors = __posteriors()
    alignment = align_ctc(posteriors, "ab", vocab)
    restored = align_from_dict(align_to_dict(alignment))
    assert restored.tokens == alignment.tokens
    assert restored.frame_spans == alignment.frame_spans


def test_score_report_roundtrip() -> None:
    score = aggregate_score(unit="character")
    score.add("u1", "abc", "abc")
    text = score_report_to_json(score)
    restored = score_report_from_json(text)
    assert restored["unit"] == "character"
    assert restored["total_errors"] == 0


def test_schema_version_is_set() -> None:
    payload = hypothesis_to_dict(greedy_decode(__posteriors(), _vocab()))
    assert payload["schema_version"] == SCHEMA_VERSION


def test_rejects_wrong_schema_version() -> None:
    with pytest.raises(SchemaError, match="schema_version"):
        hypothesis_from_dict({"schema_version": "999", "kind": "hypothesis"})


def test_rejects_wrong_kind() -> None:
    with pytest.raises(SchemaError, match="kind"):
        hypothesis_from_dict({"schema_version": SCHEMA_VERSION, "kind": "bogus"})


def test_score_report_is_valid_json() -> None:
    score = aggregate_score()
    score.add("u1", "abc", "axc")
    text = score_report_to_json(score)
    parsed = json.loads(text)
    assert parsed["schema_version"] == SCHEMA_VERSION


def __posteriors():
    import numpy as np

    return np.array(
        [
            [-5.0, 0.0, -5.0, -5.0],
            [-5.0, 0.0, -5.0, -5.0],
            [0.0, -5.0, -5.0, -5.0],
            [-5.0, -5.0, 0.0, -5.0],
            [-5.0, -5.0, 0.0, -5.0],
            [0.0, -5.0, -5.0, -5.0],
        ],
        dtype=np.float64,
    )
