"""Tests for the :mod:`wavrail.ctc._common` helpers."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.ctc._common import Hypothesis, argmax_per_frame, collapse_blanks


def test_collapse_blanks_basic() -> None:
    assert collapse_blanks([0, 1, 1, 0, 2, 2, 0, 3], blank_index=0) == [1, 2, 3]


def test_collapse_blanks_no_blanks() -> None:
    assert collapse_blanks([1, 2, 3], blank_index=0) == [1, 2, 3]


def test_collapse_blanks_consecutive_repeats_collapsed() -> None:
    assert collapse_blanks([1, 1, 1, 2], blank_index=0) == [1, 2]


def test_collapse_blanks_all_blanks() -> None:
    assert collapse_blanks([0, 0, 0], blank_index=0) == []


def test_hypothesis_length() -> None:
    h = Hypothesis(tokens=(1, 2, 3), text="abc", confidence=0.9)
    assert len(h) == 3


def test_argmax_per_frame_shape_and_dtype() -> None:
    posteriors = np.array([[0.1, 0.9], [0.7, 0.3], [0.4, 0.6]], dtype=np.float32)
    argmax = argmax_per_frame(posteriors)
    assert argmax.shape == (3,)
    assert argmax.dtype == np.int64
    assert argmax.tolist() == [1, 0, 1]


def test_argmax_per_frame_rejects_3d() -> None:
    with pytest.raises(ValueError):
        argmax_per_frame(np.zeros((2, 3, 4), dtype=np.float32))
