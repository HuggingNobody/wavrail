"""Gain staging helpers for audio buffers.

:func:`apply_gain_db` multiplies a buffer by a decibel gain and clips
the result to the representable ``[-1, 1]`` range, while
:func:`normalize_peak` rescales a buffer so its loudest sample lands
on an explicit target level. Both preserve ``float32`` dtype and the
input shape.

These helpers make no loudness-perceptual claims: there is no
A-weighting and no ITU-R BS.1770 gating, only plain sample math.
Silent buffers cannot be normalized to a nonzero peak, so they are
returned unchanged instead of producing NaNs.
"""

from __future__ import annotations

import math

import numpy as np

from wavrail.audio.energy import peak_amplitude
from wavrail.errors import AudioError
from wavrail.types import AudioArray


def apply_gain_db(samples: AudioArray, gain_db: float) -> AudioArray:
    """Scale ``samples`` by ``gain_db`` and clip the result to ``[-1, 1]``."""
    if math.isnan(gain_db) or math.isinf(gain_db):
        raise AudioError("gain_db must be finite")
    arr = np.asarray(samples, dtype=np.float32)
    if arr.size == 0:
        raise AudioError("gain requires at least one sample")
    factor = np.float32(10.0 ** (gain_db / 20.0))
    return np.clip(arr * factor, -1.0, 1.0).astype(np.float32)


def normalize_peak(samples: AudioArray, target_dbfs: float = 0.0) -> AudioArray:
    """Scale ``samples`` so the peak amplitude equals ``target_dbfs``.

    ``target_dbfs`` must be a finite value ``<= 0``. Silent input is
    returned unchanged because no finite gain can raise a zero peak.
    """
    if math.isnan(target_dbfs) or math.isinf(target_dbfs) or target_dbfs > 0.0:
        raise AudioError("target_dbfs must be a finite value <= 0")
    peak = peak_amplitude(samples)  # also rejects empty buffers
    if peak == 0.0:
        return np.asarray(samples, dtype=np.float32).copy()
    gain_db = 20.0 * math.log10(math.pow(10.0, target_dbfs / 20.0)) - 20.0 * math.log10(peak)
    return apply_gain_db(samples, gain_db)


__all__ = ["apply_gain_db", "normalize_peak"]
