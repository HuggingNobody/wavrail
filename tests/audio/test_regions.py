"""detect_silence_regions on pulse and constant fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.audio import detect_silence_regions, synthesize_pulse, synthesize_silence
from wavrail.errors import AudioError


def test_pulse_between_silence():
    samples = synthesize_pulse(0.05, 0.05, 0.2, 16000, amplitude=0.5)
    regions = detect_silence_regions(samples, threshold_db=-60.0)
    assert regions == [(0, 800), (1600, 3200)]


def test_all_silence_is_one_region():
    samples = synthesize_silence(0.1, 16000)
    assert detect_silence_regions(samples) == [(0, 1600)]


def test_constant_loud_signal_has_no_regions():
    samples = np.full(1600, 0.5, dtype=np.float32)
    assert detect_silence_regions(samples, threshold_db=-20.0) == []


def test_min_region_filters_short_gaps():
    samples = np.full(100, 0.5, dtype=np.float32)
    samples[40:42] = 0.0
    assert detect_silence_regions(samples) == [(40, 42)]
    assert detect_silence_regions(samples, min_region_samples=3) == []


def test_leading_region_only():
    samples = np.concatenate([np.zeros(10, dtype=np.float32), np.full(10, 0.5, dtype=np.float32)])
    assert detect_silence_regions(samples) == [(0, 10)]


def test_validation():
    with pytest.raises(AudioError):
        detect_silence_regions(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(AudioError):
        detect_silence_regions(np.zeros(4, dtype=np.float32), min_region_samples=0)
    with pytest.raises(AudioError):
        detect_silence_regions(np.zeros(4, dtype=np.float32), threshold_db=1.0)
