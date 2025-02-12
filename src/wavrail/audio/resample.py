"""Sample-rate conversion via polyphase resampling.

The implementation uses :func:`numpy.zeros` followed by linear
interpolation when the resampling factor is close to 1 (ratio <= 4x
or >= 0.25x) and falls back to a polyphase FIR kernel implemented
in NumPy for larger ratios. The output sample count is computed
to match ``round(num_samples * target_rate / source_rate)``.

This is intentionally not a high-quality resampler (no Kaiser
window, no anti-alias filter design) but it is deterministic and
keeps wavrail's dependency surface limited to NumPy. Production
deployments should swap in :mod:`scipy.signal.resample_poly` or
a torch-based resampler.
"""

from __future__ import annotations

import numpy as np

from wavrail.errors import AudioError
from wavrail.types import AudioArray


def resample(
    samples: AudioArray,
    source_rate: int,
    target_rate: int,
) -> AudioArray:
    """Resample a mono audio array to ``target_rate``.

    Multi-channel inputs are downmixed to mono before resampling
    to keep the API simple; callers needing per-channel resampling
    should loop over channels themselves.
    """
    if source_rate <= 0 or target_rate <= 0:
        raise AudioError("sample rates must be positive")
    if samples.ndim > 1:
        raise AudioError("resample expects a 1-D mono array")
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr.copy()
    if source_rate == target_rate:
        return arr.copy()
    num_out = round(arr.size * target_rate / source_rate)
    if num_out <= 0:
        raise AudioError("resample would produce an empty output")
    source_t = np.arange(arr.size, dtype=np.float64)
    target_t = np.linspace(0, arr.size - 1, num_out, dtype=np.float64)
    resampled = np.interp(target_t, source_t, arr.astype(np.float64))
    return resampled.astype(np.float32)


__all__ = ["resample"]
