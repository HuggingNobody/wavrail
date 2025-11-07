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


def dc_offset(samples: AudioArray) -> float:
    """Return the mean value (DC component) of ``samples``."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim != 1:
        raise AudioError("samples must be a 1-D array")
    if arr.size == 0:
        raise AudioError("dc_offset requires at least one sample")
    return float(np.mean(arr.astype(np.float64)))


def remove_dc_offset(samples: AudioArray) -> AudioArray:
    """Return ``samples`` with the mean subtracted, centring it on zero.

    ADCs and bad recordings can leave a constant bias on a signal;
    the bias inflates frame energy and shows up as a spike in the
    lowest FFT bin. Subtracting the mean is the standard remedy and
    is exactly invertible only up to the removed constant.
    """
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim != 1:
        raise AudioError("samples must be a 1-D array")
    if arr.size == 0:
        raise AudioError("remove_dc_offset requires at least one sample")
    return (arr.astype(np.float64) - dc_offset(arr)).astype(np.float32)


__all__ = ["apply_gain_db", "dc_offset", "normalize_peak", "remove_dc_offset"]
