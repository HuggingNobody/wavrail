"""alignment_summary_to_dict schema contract."""

from __future__ import annotations

import json

import pytest

from wavrail.align import Alignment
from wavrail.errors import AlignmentError
from wavrail.report import SCHEMA_VERSION, alignment_summary_to_dict


def _alignment() -> Alignment:
    return Alignment(
        tokens=(1, 2),
        frame_spans=((0, 4), (6, 10)),
        log_probabilities=(-1.0, -2.0),
        confidences=(0.9, 0.5),
    )


def test_payload_envelope_and_values():
    payload = alignment_summary_to_dict(_alignment(), frame_shift_s=0.05)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["kind"] == "alignment_summary"
    assert payload["frame_shift_s"] == 0.05
    assert payload["num_tokens"] == 2
    assert payload["duration_s"] == pytest.approx(0.5)
    assert payload["num_pauses"] == 1  # 2-frame gap = 0.1 s at this shift


def test_json_roundtrip():
    payload = alignment_summary_to_dict(_alignment(), frame_shift_s=0.02)
    assert json.loads(json.dumps(payload)) == payload


def test_empty_alignment_rejected():
    empty = Alignment(tokens=(), frame_spans=(), log_probabilities=(), confidences=())
    with pytest.raises(AlignmentError):
        alignment_summary_to_dict(empty, 0.01)
