"""Energy helpers on silence, full-scale, and constant signals."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wavrail.audio import peak_amplitude, rms_energy
from wavrail.audio.synthetic import synthesize_sine
from wavrail.errors import AudioError


def test_silence_has_zero_peak_and_rms():
    zeros = np.zeros(64, dtype=np.float32)
    assert peak_amplitude(zeros) == 0.0
    assert rms_energy(zeros) == 0.0


def test_full_scale_sine_peak_and_rms():
    tone = synthesize_sine(1000.0, 0.05, 16000, amplitude=0.8)
    assert peak_amplitude(tone) == pytest.approx(0.8, abs=1e-3)
    assert rms_energy(tone) == pytest.approx(0.8 / math.sqrt(2.0), abs=1e-2)


def test_constant_signal_peak_equals_rms():
    const = np.full(32, 0.25, dtype=np.float32)
    assert peak_amplitude(const) == pytest.approx(0.25)
    assert rms_energy(const) == pytest.approx(0.25)


def test_negative_samples_use_absolute_value():
    arr = np.array([-0.5, 0.25], dtype=np.float32)
    assert peak_amplitude(arr) == pytest.approx(0.5)


def test_multichannel_input_is_flattened():
    stereo = np.array([[-0.9, 0.1], [0.2, 0.3]], dtype=np.float32)
    assert peak_amplitude(stereo) == pytest.approx(0.9)


def test_empty_buffer_is_rejected():
    with pytest.raises(AudioError):
        peak_amplitude(np.zeros(0, dtype=np.float32))
    with pytest.raises(AudioError):
        rms_energy(np.zeros(0, dtype=np.float32))
