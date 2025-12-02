"""summarize_alignment payload, JSON-readiness, and golden key contract."""

from __future__ import annotations

import json

import pytest

from wavrail.align import Alignment, summarize_alignment
from wavrail.errors import AlignmentError


def _alignment() -> Alignment:
    return Alignment(
        tokens=(1, 2, 3),
        frame_spans=((0, 4), (6, 10), (10, 15)),
        log_probabilities=(-1.0, -2.0, -0.5),
        confidences=(0.9, 0.5, 0.7),
    )


def test_summary_payload_values():
    summary = summarize_alignment(_alignment(), 0.01, min_pause_s=0.01)
    assert summary["num_tokens"] == 3
    assert summary["total_frames"] == 15
    assert summary["duration_s"] == pytest.approx(0.15)
    assert summary["mean_confidence"] == pytest.approx((0.9 + 0.5 + 0.7) / 3)
    assert summary["min_confidence"] == pytest.approx(0.5)
    assert summary["num_pauses"] == 1
    assert summary["pauses_s"] == [[0.04, 0.06]]


def test_summary_is_json_serializable():
    payload = json.dumps(summarize_alignment(_alignment(), 0.01))
    assert json.loads(payload)["num_tokens"] == 3


def test_golden_key_contract():
    summary = summarize_alignment(_alignment(), 0.01)
    assert sorted(summary) == [
        "duration_s",
        "mean_confidence",
        "min_confidence",
        "num_pauses",
        "num_tokens",
        "pauses_s",
        "total_frames",
    ]


def test_empty_alignment_rejected():
    empty = Alignment(tokens=(), frame_spans=(), log_probabilities=(), confidences=())
    with pytest.raises(AlignmentError):
        summarize_alignment(empty, 0.01)
