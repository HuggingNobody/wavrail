"""Tests for :func:`wavrail.audio.synthesize_formant_stack`."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.audio.synthetic import synthesize_formant_stack
from wavrail.errors import AudioError


def test_single_formant_acts_like_sine() -> None:
    sig = synthesize_formant_stack([(220.0, 0.5)], 0.1, 16000)
    assert sig.shape == (1600,)
    assert sig.max() == pytest.approx(0.5, abs=1e-6)
    assert sig.min() == pytest.approx(-0.5, abs=1e-6)


def test_multiple_formants_stack_additively() -> None:
    sig = synthesize_formant_stack([(100.0, 0.4), (200.0, 0.4)], 0.05, 16000)
    expected_max = min(1.0, 0.4 + 0.4)
    assert sig.max() <= expected_max + 1e-6


def test_clipping_keeps_signal_within_bounds() -> None:
    sig = synthesize_formant_stack([(100.0, 0.9), (200.0, 0.9)], 0.05, 16000)
    assert sig.max() <= 1.0 + 1e-6
    assert sig.min() >= -1.0 - 1e-6


def test_noise_floor_adds_random_energy() -> None:
    a = synthesize_formant_stack([(100.0, 0.5)], 0.05, 16000, noise_amplitude=0.1, noise_seed=0)
    b = synthesize_formant_stack([(100.0, 0.5)], 0.05, 16000, noise_amplitude=0.0)
    assert np.abs(a - b).sum() > 0.0


def test_rejects_empty_formant_list() -> None:
    with pytest.raises(AudioError):
        synthesize_formant_stack([], 0.1, 16000)


def test_rejects_invalid_entries() -> None:
    with pytest.raises(AudioError):
        synthesize_formant_stack([(0.0, 0.5)], 0.1, 16000)
    with pytest.raises(AudioError):
        synthesize_formant_stack([(100.0, 1.5)], 0.1, 16000)
    with pytest.raises(AudioError):
        synthesize_formant_stack([(100.0,)], 0.1, 16000)  # type: ignore[arg-type]
