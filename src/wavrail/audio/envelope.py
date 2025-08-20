"""Fade-in / fade-out envelope shaping.

Clicks at buffer boundaries are the most common artifact when
synthetic fixtures are concatenated. :func:`apply_fade` multiplies a
buffer by raised-cosine ramps so its edges start and end near zero.
The envelope is computed in float64 and applied along the last
(time) axis, so both mono ``(num_samples,)`` and multi-channel
``(num_channels, num_samples)`` layouts are supported.

When the two ramps overlap, both gains multiply, which keeps the
envelope continuous instead of jumping at the crossover point.
"""

from __future__ import annotations

import numpy as np

from wavrail.errors import AudioError
from wavrail.types import AudioArray


def _raised_cosine(num_samples: int, rising: bool) -> np.ndarray:
    if num_samples <= 1:
        return np.zeros(max(num_samples, 0), dtype=np.float64)
    n = np.arange(num_samples, dtype=np.float64)
    gain = 0.5 * (1.0 - np.cos(np.pi * n / (num_samples - 1)))
    return gain if rising else gain[::-1].copy()


def apply_fade(
    samples: AudioArray,
    sample_rate: int,
    fade_in_s: float = 0.0,
    fade_out_s: float = 0.0,
) -> AudioArray:
    """Apply raised-cosine fades of the given durations to ``samples``.

    The output keeps the input shape and ``float32`` dtype. Raises
    :class:`AudioError` for negative fades, non-positive sample
    rates, empty input, or either fade longer than the signal.
    Overlapping ramps are allowed and multiply into each other.
    """
    if sample_rate <= 0:
        raise AudioError("sample_rate must be positive")
    if fade_in_s < 0.0 or fade_out_s < 0.0:
        raise AudioError("fade durations must be non-negative")
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 0 or arr.shape[-1] == 0:
        raise AudioError("apply_fade requires at least one sample")
    length = arr.shape[-1]
    n_in = round(fade_in_s * sample_rate)
    n_out = round(fade_out_s * sample_rate)
    if n_in > length or n_out > length:
        raise AudioError("fade duration exceeds the signal length")
    envelope = np.ones(length, dtype=np.float64)
    if n_in > 0:
        envelope[:n_in] *= _raised_cosine(n_in, rising=True)
    if n_out > 0:
        envelope[length - n_out :] *= _raised_cosine(n_out, rising=False)
    return (arr.astype(np.float64) * envelope).astype(np.float32)


__all__ = ["apply_fade"]
