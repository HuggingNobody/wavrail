"""Spectral centroid and bandwidth on controlled spectra."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.audio import synthesize_sine
from wavrail.errors import FeatureError
from wavrail.features import (
    frame_signal,
    hann_window,
    power_spectrum,
    spectral_bandwidth,
    spectral_centroid,
)
from wavrail.types import FrameConfig

BIN_HZ = 16000 / 128  # 65-bin spectra correspond to n_fft = 128


def _single_bin(bin_index: int, num_frames: int = 3) -> np.ndarray:
    spectra = np.zeros((num_frames, 65), dtype=np.float32)
    spectra[:, bin_index] = 1.0
    return spectra


def test_centroid_of_single_bin_is_bin_frequency():
    centroid = spectral_centroid(_single_bin(20), 16000)
    assert centroid.shape == (3,)
    assert centroid.dtype == np.float32
    assert np.allclose(centroid, 20 * BIN_HZ, rtol=1e-5)


def test_bandwidth_of_single_bin_is_zero():
    assert np.allclose(spectral_bandwidth(_single_bin(10), 16000), 0.0, atol=1e-2)


def test_two_equal_bins_centroid_is_midpoint():
    spectra = np.zeros((1, 65), dtype=np.float32)
    spectra[0, 10] = 0.5
    spectra[0, 30] = 0.5
    centroid = spectral_centroid(spectra, 16000)
    assert centroid[0] == pytest.approx(20 * BIN_HZ, rel=1e-5)


def test_silence_returns_zero():
    spectra = np.zeros((2, 65), dtype=np.float32)
    assert np.all(spectral_centroid(spectra, 16000) == 0.0)
    assert np.all(spectral_bandwidth(spectra, 16000) == 0.0)


def test_centroid_tracks_sine_frequency():
    tone = synthesize_sine(2000.0, 0.05, 16000, amplitude=0.8)
    frames = frame_signal(tone, FrameConfig(frame_length=400, hop_length=160))
    spectra = power_spectrum(frames, fft_size=512, window=hann_window(400))
    centroid = spectral_centroid(spectra, 16000)
    assert centroid.shape == (spectra.shape[0],)
    assert np.all(np.abs(centroid - 2000.0) < 100.0)


def test_rejects_bad_input():
    with pytest.raises(FeatureError):
        spectral_centroid(np.zeros((2, 1), dtype=np.float32), 16000)
    with pytest.raises(FeatureError):
        spectral_centroid(np.zeros((2, 65), dtype=np.float32), 0)
    with pytest.raises(FeatureError):
        spectral_bandwidth(np.zeros(65, dtype=np.float32), 16000)
    negative = np.zeros((1, 65), dtype=np.float32)
    negative[0, 5] = -1.0
    with pytest.raises(FeatureError):
        spectral_bandwidth(negative, 16000)
