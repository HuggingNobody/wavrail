"""synthesize_multitone additivity, Nyquist guard, and determinism."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.audio import synthesize_multitone, synthesize_sine
from wavrail.errors import AudioError


def test_single_tone_matches_scaled_sine():
    mix = synthesize_multitone([440.0], 0.05, 16000, amplitude=0.6)
    tone = synthesize_sine(440.0, 0.05, 16000, amplitude=0.6)
    assert np.allclose(mix, tone, atol=1e-6)


def test_two_tones_add_linearly():
    a = synthesize_multitone([300.0], 0.05, 16000, amplitude=0.4)
    b = synthesize_multitone([700.0], 0.05, 16000, amplitude=0.4)
    mix = synthesize_multitone([300.0, 700.0], 0.05, 16000, amplitude=0.4)
    assert np.allclose(mix, (a + b) / 2.0, atol=1e-6)


def test_nyquist_violation_rejected():
    with pytest.raises(AudioError):
        synthesize_multitone([8000.0], 0.01, 16000)  # exactly at Nyquist
    with pytest.raises(AudioError):
        synthesize_multitone([9000.0], 0.01, 16000)


def test_empty_and_bad_params_rejected():
    with pytest.raises(AudioError):
        synthesize_multitone([], 0.01, 16000)
    with pytest.raises(AudioError):
        synthesize_multitone([-440.0], 0.01, 16000)
    with pytest.raises(AudioError):
        synthesize_multitone([440.0], 0.01, 16000, amplitude=1.5)


def test_mix_stays_within_unit_range():
    mix = synthesize_multitone([200.0, 400.0, 800.0], 0.05, 16000, amplitude=1.0)
    assert np.max(np.abs(mix)) <= 1.0
    assert mix.dtype == np.float32
