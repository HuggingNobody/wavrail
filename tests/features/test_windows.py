"""Tests for the window function helpers."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.errors import FeatureError
from wavrail.features.windows import (
    blackman_window,
    hamming_window,
    hann_window,
    rectangular_window,
)


def test_hann_window_shape_and_dtype() -> None:
    w = hann_window(16)
    assert w.shape == (16,)
    assert w.dtype == np.float32
    assert w[0] == pytest.approx(0.0, abs=1e-7)
    assert w.max() > 0.97
    assert w.max() <= 1.0


def test_hamming_window_endpoints() -> None:
    w = hamming_window(16)
    assert w[0] == pytest.approx(0.08, abs=1e-6)
    assert w.max() > 0.97
    assert w.max() <= 1.0


def test_blackman_window_endpoints() -> None:
    w = blackman_window(16)
    assert w[0] == pytest.approx(0.0, abs=1e-7)
    assert w.max() > 0.97
    assert w.max() <= 1.0


def test_blackman_rejects_short_windows() -> None:
    with pytest.raises(FeatureError):
        blackman_window(2)


def test_rectangular_window_is_all_ones() -> None:
    w = rectangular_window(10)
    np.testing.assert_array_equal(w, np.ones(10, dtype=np.float32))


def test_symmetric_and_periodic_hann_differ() -> None:
    sym = hann_window(8, symmetric=True)
    per = hann_window(8, symmetric=False)
    assert not np.array_equal(sym, per)
