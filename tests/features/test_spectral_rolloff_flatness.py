"""Rolloff percentile behaviour and flatness extremes."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.errors import FeatureError
from wavrail.features import spectral_flatness, spectral_rolloff

BIN_HZ = 16000 / 128


def test_rolloff_of_low_band_spectrum():
    spectra = np.zeros((1, 65), dtype=np.float32)
    spectra[0, 1:11] = 1.0  # equal energy in bins 1..10
    rolloff = spectral_rolloff(spectra, 16000, percent=0.85)
    # cumulative energy first reaches 85% of the total inside bin 9
    assert rolloff[0] == pytest.approx(9 * BIN_HZ, rel=1e-6)


def test_rolloff_percentiles_are_monotone():
    rng = np.random.default_rng(7)
    spectra = rng.random((5, 65)).astype(np.float32)
    low = spectral_rolloff(spectra, 16000, percent=0.25)
    high = spectral_rolloff(spectra, 16000, percent=0.95)
    assert np.all(low <= high + 1e-6)


def test_rolloff_percent_validation():
    spectra = np.ones((1, 65), dtype=np.float32)
    with pytest.raises(FeatureError):
        spectral_rolloff(spectra, 16000, percent=0.0)
    with pytest.raises(FeatureError):
        spectral_rolloff(spectra, 16000, percent=1.5)
    assert np.all(spectral_rolloff(spectra, 16000, percent=1.0) >= 0.0)


def test_flatness_of_uniform_spectrum_is_one():
    spectra = np.full((2, 65), 0.5, dtype=np.float32)
    flat = spectral_flatness(spectra, 16000)
    assert np.allclose(flat, 1.0, atol=1e-5)


def test_flatness_of_tone_is_near_zero():
    spectra = np.zeros((1, 65), dtype=np.float32)
    spectra[0, 32] = 1.0
    assert spectral_flatness(spectra, 16000)[0] < 1e-3


def test_flatness_silence_is_zero():
    spectra = np.zeros((3, 65), dtype=np.float32)
    assert np.all(spectral_flatness(spectra, 16000) == 0.0)
