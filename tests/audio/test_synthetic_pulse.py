"""Tests for :func:`wavrail.audio.synthesize_pulse`."""

from __future__ import annotations

import pytest

from wavrail.audio import synthesize_pulse
from wavrail.errors import AudioError


def test_pulse_placed_at_start() -> None:
    sig = synthesize_pulse(start_s=0.0, duration_s=0.01, total_duration_s=0.1, sample_rate=16000)
    assert sig.shape == (1600,)
    assert sig[:160].sum() == pytest.approx(160.0)
    assert sig[160:].sum() == 0.0


def test_pulse_offset_from_start() -> None:
    sig = synthesize_pulse(start_s=0.05, duration_s=0.02, total_duration_s=0.1, sample_rate=16000)
    assert sig[:800].sum() == 0.0
    assert sig[800:1120].sum() == pytest.approx(320.0)
    assert sig[1120:].sum() == 0.0


def test_amplitude_zero_produces_silence() -> None:
    sig = synthesize_pulse(0.0, 0.1, 0.1, 16000, amplitude=0.0)
    assert sig.sum() == 0.0


def test_rejects_out_of_range_pulse() -> None:
    with pytest.raises(AudioError):
        synthesize_pulse(-0.01, 0.1, 0.1, 16000)
    with pytest.raises(AudioError):
        synthesize_pulse(0.0, 0.2, 0.1, 16000)
    with pytest.raises(AudioError):
        synthesize_pulse(0.0, 0.0, 0.1, 16000)


def test_rejects_invalid_amplitude() -> None:
    with pytest.raises(AudioError):
        synthesize_pulse(0.0, 0.01, 0.1, 16000, amplitude=-0.1)
    with pytest.raises(AudioError):
        synthesize_pulse(0.0, 0.01, 0.1, 16000, amplitude=1.1)
