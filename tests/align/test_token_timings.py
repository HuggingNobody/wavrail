"""token_timings frame-to-second conversions."""

from __future__ import annotations

import pytest

from wavrail.align import Alignment, TokenTiming, token_timings
from wavrail.errors import AlignmentError


def _alignment() -> Alignment:
    return Alignment(
        tokens=(1, 2, 3),
        frame_spans=((0, 4), (6, 10), (10, 15)),
        log_probabilities=(-1.0, -2.0, -0.5),
        confidences=(0.9, 0.5, 0.7),
    )


def test_timings_convert_frames_to_seconds():
    timings = token_timings(_alignment(), frame_shift_s=0.01)
    assert len(timings) == 3
    first = timings[0]
    assert isinstance(first, TokenTiming)
    assert first.index == 0
    assert first.token_id == 1
    assert first.start_s == pytest.approx(0.0)
    assert first.end_s == pytest.approx(0.04)
    assert first.duration_s == pytest.approx(0.04)
    assert timings[2].end_s == pytest.approx(0.15)


def test_confidence_is_carried_through():
    timings = token_timings(_alignment(), 0.01)
    assert timings[1].confidence == pytest.approx(0.5)


def test_frame_fields_are_preserved():
    timings = token_timings(_alignment(), 0.02)
    assert (timings[1].start_frame, timings[1].end_frame) == (6, 10)
    assert timings[1].start_s == pytest.approx(0.12)


def test_empty_alignment_yields_empty_list():
    empty = Alignment(tokens=(), frame_spans=(), log_probabilities=(), confidences=())
    assert token_timings(empty, 0.01) == []


def test_rejects_non_positive_shift():
    with pytest.raises(AlignmentError):
        token_timings(_alignment(), 0.0)
