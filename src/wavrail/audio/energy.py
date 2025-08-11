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


__all__ = ["peak_amplitude", "rms_energy"]
