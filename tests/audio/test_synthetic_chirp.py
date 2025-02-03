"""Tests for :func:`wavrail.audio.synthesize_chirp`."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.audio import synthesize_chirp
from wavrail.errors import AudioError


def test_shape_matches_duration_and_rate() -> None:
    sig = synthesize_chirp(50.0, 1000.0, 0.1, 16000)
    assert sig.shape == (1600,)
    assert sig.dtype == np.float32


def test_amplitude_bounded() -> None:
    sig = synthesize_chirp(10.0, 500.0, 0.2, 16000, amplitude=0.9)
    assert sig.max() <= 0.9 + 1e-6
    assert sig.min() >= -0.9 - 1e-6


def test_deterministic_for_same_arguments() -> None:
    a = synthesize_chirp(20.0, 2000.0, 0.05, 16000)
    b = synthesize_chirp(20.0, 2000.0, 0.05, 16000)
    np.testing.assert_array_equal(a, b)


def test_zero_duration_rejected() -> None:
    with pytest.raises(AudioError):
        synthesize_chirp(10.0, 100.0, 0.0, 16000)


def test_rejects_nonpositive_frequencies() -> None:
    with pytest.raises(AudioError):
        synthesize_chirp(0.0, 100.0, 0.1, 16000)
    with pytest.raises(AudioError):
        synthesize_chirp(100.0, -10.0, 0.1, 16000)
