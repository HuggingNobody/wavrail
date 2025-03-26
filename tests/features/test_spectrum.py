"""Tests for :func:`wavrail.features.power_spectrum`."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.errors import FeatureError
from wavrail.features.spectrum import power_spectrum


def test_default_fft_size_matches_frame_length() -> None:
    frames = np.eye(4, dtype=np.float32)
    spectrum = power_spectrum(frames, fft_size=4)
    assert spectrum.shape == (4, 3)
    assert spectrum.dtype == np.float32


def test_padded_fft_produces_extra_bins() -> None:
    frames = np.eye(4, dtype=np.float32)
    spectrum = power_spectrum(frames, fft_size=16)
    assert spectrum.shape == (4, 9)


def test_window_changes_spectrum() -> None:
    frames = np.ones((1, 8), dtype=np.float32)
    rect = power_spectrum(frames, fft_size=8)
    hann = power_spectrum(frames, fft_size=8, window=np.hanning(8).astype(np.float32))
    assert rect[0, 0] > hann[0, 0]


def test_rejects_bad_fft_size() -> None:
    frames = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(FeatureError):
        power_spectrum(frames, fft_size=0)


def test_rejects_wrong_window_length() -> None:
    frames = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(FeatureError):
        power_spectrum(frames, fft_size=4, window=np.ones(3, dtype=np.float32))
