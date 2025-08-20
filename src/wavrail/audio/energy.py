"""Amplitude statistics for audio buffers.

These helpers quantify "how loud is this buffer" without making any
perceptual claim: :func:`peak_amplitude` reports the largest
absolute sample, and :func:`rms_energy` reports the root-mean-square
amplitude. Both flatten multi-channel input, matching the policy of
the quality heuristics in :mod:`wavrail.audio.quality`.

Unlike the quality heuristics, empty buffers are rejected: amplitude
statistics are undefined when there is not a single sample.
"""

from __future__ import annotations

import math

import numpy as np

from wavrail.errors import AudioError
from wavrail.types import AudioArray


def _flat_nonempty(samples: AudioArray) -> AudioArray:
    flat = np.asarray(samples, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        raise AudioError("energy statistics require at least one sample")
    return flat


def peak_amplitude(samples: AudioArray) -> float:
    """Return the largest absolute sample value in ``samples``."""
    return float(np.max(np.abs(_flat_nonempty(samples))))


def rms_energy(samples: AudioArray) -> float:
    """Return the root-mean-square amplitude of ``samples``.

    The accumulation happens in float64 so long buffers of small
    values do not lose precision; the return value is a plain
    Python float.
    """
    flat = _flat_nonempty(samples).astype(np.float64)
    return float(np.sqrt(np.mean(flat * flat)))


def crest_factor(samples: AudioArray) -> float:
    """Return the peak-to-RMS ratio of ``samples``.

    The crest factor separates impulsive signals (high ratio) from
    steady tones (ratio near ``sqrt(2)`` for a sine). A silent
    buffer has no meaningful ratio; ``0.0`` is returned instead of
    dividing by zero.
    """
    rms = rms_energy(samples)
    if rms == 0.0:
        return 0.0
    return peak_amplitude(samples) / rms


def to_dbfs(value: float) -> float:
    """Convert a linear amplitude in ``(0, 1]`` to decibels relative to full scale."""
    if not 0.0 < value <= 1.0:
        raise AudioError("to_dbfs requires a linear amplitude in (0, 1]")
    return float(20.0 * math.log10(value))


def from_dbfs(db: float) -> float:
    """Convert a level in ``(-inf, 0]`` dBFS back to a linear amplitude.

    ``-inf`` maps to ``0.0``; positive or NaN levels are rejected
    because full scale is the ceiling of the representation.
    """
    if math.isnan(db) or db > 0.0:
        raise AudioError("from_dbfs requires a level <= 0 dBFS")
    return float(10.0 ** (db / 20.0))


__all__ = ["crest_factor", "from_dbfs", "peak_amplitude", "rms_energy", "to_dbfs"]
