"""Tests for :func:`wavrail.audio.synthesize_sine`."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.audio import synthesize_sine
from wavrail.errors import AudioError


def test_shape_matches_duration_and_rate() -> None:
    sig = synthesize_sine(440.0, 0.1, 16000)
    assert sig.shape == (1600,)
    assert sig.dtype == np.float32


def test_amplitude_scales_signal() -> None:
    sig = synthesize_sine(100.0, 0.5, 16000, amplitude=0.8)
    assert sig.max() == pytest.approx(0.8, abs=1e-6)
    assert sig.min() == pytest.approx(-0.8, abs=1e-6)


def test_deterministic_for_same_arguments() -> None:
    a = synthesize_sine(220.0, 0.05, 16000)
    b = synthesize_sine(220.0, 0.05, 16000)
    np.testing.assert_array_equal(a, b)


def test_phase_shifts_zero_crossing() -> None:
    base = synthesize_sine(100.0, 0.1, 16000, phase_rad=0.0)
    shifted = synthesize_sine(100.0, 0.1, 16000, phase_rad=np.pi / 2)
    assert base[0] == pytest.approx(0.0, abs=1e-6)
    assert shifted[0] == pytest.approx(0.5, abs=1e-6)


def test_rejects_invalid_arguments() -> None:
    with pytest.raises(AudioError):
        synthesize_sine(0.0, 0.1, 16000)
    with pytest.raises(AudioError):
        synthesize_sine(220.0, -0.1, 16000)
    with pytest.raises(AudioError):
        synthesize_sine(220.0, 0.1, 0)
    with pytest.raises(AudioError):
        synthesize_sine(220.0, 0.1, 16000, amplitude=1.5)
