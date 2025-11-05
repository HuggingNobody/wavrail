"""trim_silence boundaries, threshold semantics, and validation."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.audio import synthesize_pulse, synthesize_silence, trim_silence
from wavrail.errors import AudioError


def test_trims_exact_pulse_boundaries():
    samples = synthesize_pulse(0.05, 0.05, 0.2, 16000, amplitude=0.5)
    trimmed = trim_silence(samples, threshold_db=-60.0)
    assert trimmed.shape == (800,)
    np.testing.assert_allclose(trimmed, 0.5, atol=1e-6)


def test_all_silence_trims_to_empty():
    assert trim_silence(synthesize_silence(0.1, 16000)).size == 0


def test_threshold_above_signal_removes_everything():
    samples = synthesize_pulse(0.05, 0.05, 0.2, 16000, amplitude=0.5)  # -6 dBFS
    assert trim_silence(samples, threshold_db=-3.0).size == 0


def test_sample_exactly_at_threshold_counts_as_quiet():
    samples = np.array([0.0, 0.5, 0.0], dtype=np.float32)
    at_threshold = 20.0 * np.log10(0.5)  # |0.5| == limit -> quiet -> all trimmed
    assert trim_silence(samples, threshold_db=float(at_threshold)).size == 0
    below = trim_silence(samples, threshold_db=float(at_threshold) - 0.01)
    np.testing.assert_allclose(below, np.array([0.5], dtype=np.float32))


def test_input_is_not_mutated():
    samples = synthesize_pulse(0.05, 0.05, 0.2, 16000, amplitude=0.5)
    before = samples.copy()
    trim_silence(samples)
    np.testing.assert_array_equal(samples, before)


def test_validation():
    with pytest.raises(AudioError):
        trim_silence(np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(AudioError):
        trim_silence(np.zeros(8, dtype=np.float32), threshold_db=1.0)
