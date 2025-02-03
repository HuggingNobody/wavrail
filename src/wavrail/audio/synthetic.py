"""Deterministic synthetic waveform generators.

These helpers produce reproducible test fixtures for the rest of
the wavrail package. They accept an explicit ``seed`` argument and
seed an internal :class:`numpy.random.Generator` instance; the
default seed is ``0`` so that tests run deterministically without
having to thread a seed through every call.

All generated signals have shape ``(num_samples,)`` and dtype
``float32`` in the range ``[-1, 1]``. Multi-tone helpers stack the
tones additively rather than averaging so callers can reason about
amplitudes explicitly.
"""

from __future__ import annotations

import numpy as np

from wavrail.errors import AudioError
from wavrail.types import AudioArray


def _check_positive(name: str, value: float) -> None:
    if value <= 0:
        raise AudioError(f"{name} must be positive (got {value})")


def _time_axis(num_samples: int, sample_rate: int) -> AudioArray:
    return np.arange(num_samples, dtype=np.float32) / float(sample_rate)


def synthesize_sine(
    frequency_hz: float,
    duration_s: float,
    sample_rate: int,
    amplitude: float = 0.5,
    phase_rad: float = 0.0,
) -> AudioArray:
    """Generate a pure sine tone."""
    _check_positive("frequency_hz", frequency_hz)
    _check_positive("duration_s", duration_s)
    _check_positive("sample_rate", sample_rate)
    if not 0.0 <= amplitude <= 1.0:
        raise AudioError("amplitude must be in [0, 1]")
    num_samples = round(duration_s * sample_rate)
    t = _time_axis(num_samples, sample_rate)
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * t + phase_rad).astype(np.float32)


def synthesize_chirp(
    start_hz: float,
    end_hz: float,
    duration_s: float,
    sample_rate: int,
    amplitude: float = 0.5,
) -> AudioArray:
    """Generate a linear-frequency chirp from ``start_hz`` to ``end_hz``."""
    _check_positive("duration_s", duration_s)
    _check_positive("sample_rate", sample_rate)
    if start_hz <= 0 or end_hz <= 0:
        raise AudioError("start_hz and end_hz must be positive")
    if not 0.0 <= amplitude <= 1.0:
        raise AudioError("amplitude must be in [0, 1]")
    num_samples = round(duration_s * sample_rate)
    t = _time_axis(num_samples, sample_rate)
    k = (end_hz - start_hz) / duration_s
    phase = 2.0 * np.pi * (start_hz * t + 0.5 * k * t * t)
    return (amplitude * np.sin(phase)).astype(np.float32)


def synthesize_pulse(
    start_s: float,
    duration_s: float,
    total_duration_s: float,
    sample_rate: int,
    amplitude: float = 1.0,
) -> AudioArray:
    """Generate a rectangular pulse sitting on a silent baseline.

    The output length is ``round(total_duration_s * sample_rate)``.
    Samples inside ``[start_s, start_s + duration_s]`` take the
    requested amplitude; everything else is zero.
    """
    _check_positive("total_duration_s", total_duration_s)
    _check_positive("sample_rate", sample_rate)
    if duration_s <= 0:
        raise AudioError("duration_s must be positive")
    if start_s < 0:
        raise AudioError("start_s must be non-negative")
    if start_s + duration_s > total_duration_s:
        raise AudioError("start_s + duration_s exceeds total_duration_s")
    if not 0.0 <= amplitude <= 1.0:
        raise AudioError("amplitude must be in [0, 1]")
    total = round(total_duration_s * sample_rate)
    pulse = np.zeros(total, dtype=np.float32)
    begin = round(start_s * sample_rate)
    end = begin + round(duration_s * sample_rate)
    pulse[begin:end] = amplitude
    return pulse


__all__ = ["synthesize_chirp", "synthesize_pulse", "synthesize_sine"]
