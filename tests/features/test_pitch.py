"""Autocorrelation pitch estimates on synthetic fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.errors import FeatureError
from wavrail.features import PitchEstimate, estimate_pitch_acf


def _sine(freq: float, num_samples: int = 800, sample_rate: int = 8000) -> np.ndarray:
    t = np.arange(num_samples, dtype=np.float64) / sample_rate
    return np.sin(2.0 * np.pi * freq * t)


def test_sine_pitch_is_recovered():
    estimate = estimate_pitch_acf(_sine(120.0), 8000)
    assert isinstance(estimate, PitchEstimate)
    assert estimate.frequency_hz == pytest.approx(120.0, rel=0.02)
    assert estimate.confidence > 0.8


def test_higher_sine_pitch():
    estimate = estimate_pitch_acf(_sine(250.0), 8000)
    assert estimate.frequency_hz == pytest.approx(250.0, rel=0.02)
    assert estimate.confidence > 0.8


def test_noise_has_low_confidence():
    rng = np.random.default_rng(3)
    noise = rng.uniform(-1.0, 1.0, size=800)
    estimate = estimate_pitch_acf(noise, 8000)
    assert estimate.confidence < 0.5


def test_silence_is_unvoiced():
    estimate = estimate_pitch_acf(np.zeros(800), 8000)
    assert estimate.frequency_hz == 0.0
    assert estimate.confidence == 0.0


def test_validation():
    with pytest.raises(FeatureError):
        estimate_pitch_acf(_sine(120.0), 0)
    with pytest.raises(FeatureError):
        estimate_pitch_acf(_sine(120.0), 8000, freq_range=(400.0, 60.0))
    with pytest.raises(FeatureError):
        estimate_pitch_acf(np.zeros(10), 8000)  # too short for lags up to 134
