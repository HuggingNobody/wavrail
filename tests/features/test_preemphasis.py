"""Tests for :func:`wavrail.features.pre_emphasize`."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.errors import FeatureError
from wavrail.features.preemphasis import pre_emphasize


def test_zero_coefficient_is_identity() -> None:
    samples = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32)
    np.testing.assert_array_equal(pre_emphasize(samples, coefficient=0.0), samples)


def test_default_coefficient_high_passes() -> None:
    samples = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    out = pre_emphasize(samples)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(1.0 - 0.97)
    assert out[2] == pytest.approx(1.0 - 0.97)
    assert out[3] == pytest.approx(1.0 - 0.97)


def test_empty_signal_returns_empty() -> None:
    out = pre_emphasize(np.zeros(0, dtype=np.float32))
    assert out.shape == (0,)
    assert out.dtype == np.float32


def test_rejects_invalid_coefficient() -> None:
    samples = np.zeros(4, dtype=np.float32)
    with pytest.raises(FeatureError):
        pre_emphasize(samples, coefficient=-0.1)
    with pytest.raises(FeatureError):
        pre_emphasize(samples, coefficient=1.0)
