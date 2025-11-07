"""DC offset measurement and removal."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.audio import dc_offset, remove_dc_offset
from wavrail.errors import AudioError


def test_dc_offset_measures_mean():
    samples = np.array([1.0, 1.0, -0.5, -0.5, 0.5], dtype=np.float32)
    assert dc_offset(samples) == pytest.approx(0.3, abs=1e-6)


def test_removal_centres_signal():
    samples = (np.arange(100, dtype=np.float32) / 100.0) * 0.5 + 0.3
    centered = remove_dc_offset(samples)
    assert centered.dtype == np.float32
    assert dc_offset(centered) == pytest.approx(0.0, abs=1e-6)


def test_removal_preserves_shape_differences():
    samples = np.array([0.1, 0.3, -0.2, 0.4], dtype=np.float32)
    centered = remove_dc_offset(samples)
    np.testing.assert_allclose(np.diff(centered), np.diff(samples), atol=1e-6)


def test_zero_mean_input_unchanged():
    samples = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32)
    np.testing.assert_allclose(remove_dc_offset(samples), samples, atol=1e-6)


def test_validation():
    with pytest.raises(AudioError):
        dc_offset(np.zeros(0, dtype=np.float32))
    with pytest.raises(AudioError):
        dc_offset(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(AudioError):
        remove_dc_offset(np.zeros(0, dtype=np.float32))
