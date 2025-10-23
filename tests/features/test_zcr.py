"""Zero-crossing rate on controlled frames."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.errors import FeatureError
from wavrail.features import zero_crossing_rate


def test_constant_frame_has_no_crossings():
    frames = np.ones((3, 64), dtype=np.float32)
    zcr = zero_crossing_rate(frames)
    assert zcr.shape == (3,)
    assert zcr.dtype == np.float32
    np.testing.assert_allclose(zcr, np.zeros(3), atol=1e-7)


def test_alternating_frame_is_one():
    frames = np.tile(np.array([1.0, -1.0], dtype=np.float32), (1, 32))
    zcr = zero_crossing_rate(frames)
    assert zcr[0] == pytest.approx(1.0)


def test_sine_rate_matches_frequency():
    freq, sample_rate = 1000.0, 16000
    t = np.arange(1600, dtype=np.float64) / sample_rate
    frames = np.sin(2.0 * np.pi * freq * t).astype(np.float32).reshape(1, -1)
    zcr = zero_crossing_rate(frames)
    assert zcr[0] == pytest.approx(2.0 * freq / sample_rate, abs=0.01)


def test_zero_counts_as_positive():
    frames = np.array([[0.0, 0.0, -1.0, -1.0]], dtype=np.float32)
    zcr = zero_crossing_rate(frames)
    assert zcr[0] == pytest.approx(1.0 / 3.0)


def test_validation():
    with pytest.raises(FeatureError):
        zero_crossing_rate(np.zeros(16, dtype=np.float32))
    with pytest.raises(FeatureError):
        zero_crossing_rate(np.zeros((2, 1), dtype=np.float32))
