"""Peak normalisation and dB gain application."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.audio import apply_gain_db, normalize_peak, peak_amplitude
from wavrail.errors import AudioError


def test_normalize_raises_quiet_signal_to_full_scale():
    quiet = np.full(16, 0.25, dtype=np.float32)
    boosted = normalize_peak(quiet, target_dbfs=0.0)
    assert peak_amplitude(boosted) == pytest.approx(1.0, rel=1e-5)


def test_normalize_to_negative_target():
    loud = np.full(16, 0.8, dtype=np.float32)
    scaled = normalize_peak(loud, target_dbfs=-6.0206)
    assert peak_amplitude(scaled) == pytest.approx(0.5, rel=1e-3)


def test_gain_scales_and_clips():
    arr = np.array([0.1, -0.2], dtype=np.float32)
    gained = apply_gain_db(arr, 20.0)
    assert gained[0] == pytest.approx(1.0, rel=1e-5)
    assert gained[1] == -1.0  # -2.0 clipped to the representable range


def test_zero_db_gain_is_identity():
    arr = np.linspace(-0.5, 0.5, 32, dtype=np.float32)
    assert np.array_equal(apply_gain_db(arr, 0.0), arr)


def test_silence_normalizes_to_silence():
    zeros = np.zeros(8, dtype=np.float32)
    out = normalize_peak(zeros)
    assert np.array_equal(out, zeros)


def test_validation_errors():
    with pytest.raises(AudioError):
        normalize_peak(np.zeros(4, dtype=np.float32), target_dbfs=1.0)
    with pytest.raises(AudioError):
        normalize_peak(np.zeros(4, dtype=np.float32), target_dbfs=float("inf"))
    with pytest.raises(AudioError):
        apply_gain_db(np.zeros(0, dtype=np.float32), 0.0)
    with pytest.raises(AudioError):
        apply_gain_db(np.zeros(4, dtype=np.float32), float("nan"))
    with pytest.raises(AudioError):
        normalize_peak(np.zeros(0, dtype=np.float32))
