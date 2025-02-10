"""Tests for :func:`wavrail.audio.synthesize_noise`."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.audio.synthetic import synthesize_noise
from wavrail.errors import AudioError


def test_shape_matches_duration_and_rate() -> None:
    sig = synthesize_noise(0.1, 16000)
    assert sig.shape == (1600,)
    assert sig.dtype == np.float32


def test_default_amplitude_bounds_samples() -> None:
    sig = synthesize_noise(0.5, 16000)
    assert sig.max() <= 0.1 + 1e-7
    assert sig.min() >= -0.1 - 1e-7


def test_seed_makes_output_deterministic() -> None:
    a = synthesize_noise(0.1, 16000, seed=42)
    b = synthesize_noise(0.1, 16000, seed=42)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_produce_different_outputs() -> None:
    a = synthesize_noise(0.1, 16000, seed=1)
    b = synthesize_noise(0.1, 16000, seed=2)
    assert not np.array_equal(a, b)


def test_rejects_invalid_arguments() -> None:
    with pytest.raises(AudioError):
        synthesize_noise(0.0, 16000)
    with pytest.raises(AudioError):
        synthesize_noise(0.1, 16000, amplitude=1.5)
