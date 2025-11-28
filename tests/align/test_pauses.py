"""detect_pauses on gapped and touching alignment spans."""

from __future__ import annotations

import pytest

from wavrail.align import Alignment, detect_pauses
from wavrail.errors import AlignmentError


def _alignment() -> Alignment:
    return Alignment(
        tokens=(1, 2, 3),
        frame_spans=((0, 4), (6, 10), (10, 15)),
        log_probabilities=(-1.0, -2.0, -0.5),
        confidences=(0.9, 0.5, 0.7),
    )


def test_gap_between_tokens_is_pause():
    pauses = detect_pauses(_alignment(), 0.01, min_pause_s=0.01)
    assert len(pauses) == 1
    assert pauses[0][0] == pytest.approx(0.04)
    assert pauses[0][1] == pytest.approx(0.06)


def test_min_pause_filters_short_gaps():
    assert detect_pauses(_alignment(), 0.01, min_pause_s=0.05) == []


def test_touching_spans_have_no_pause():
    alignment = Alignment(
        tokens=(1, 2),
        frame_spans=((0, 5), (5, 9)),
        log_probabilities=(-1.0, -1.0),
        confidences=(0.9, 0.9),
    )
    assert detect_pauses(alignment, 0.01) == []


def test_single_token_has_no_pauses():
    alignment = Alignment(
        tokens=(1,), frame_spans=((2, 7),), log_probabilities=(-0.5,), confidences=(0.8,)
    )
    assert detect_pauses(alignment, 0.01) == []


def test_validation():
    with pytest.raises(AlignmentError):
        detect_pauses(_alignment(), 0.0)
    with pytest.raises(AlignmentError):
        detect_pauses(_alignment(), 0.01, min_pause_s=-0.1)
