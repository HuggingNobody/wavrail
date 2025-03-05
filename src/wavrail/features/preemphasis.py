"""Pre-emphasis filter.

Pre-emphasis boosts high-frequency energy to compensate for the
natural spectral tilt of voiced speech. The default coefficient
(``0.97``) follows the convention used by HTK and Kaldi.
"""

from __future__ import annotations

import numpy as np

from wavrail.errors import FeatureError
from wavrail.types import AudioArray


def pre_emphasize(samples: AudioArray, coefficient: float = 0.97) -> AudioArray:
    """Apply a one-pole high-pass filter ``y[n] = x[n] - coef * x[n-1]``.

    The output has the same length as the input; the first sample is
    returned unchanged because no preceding sample exists.
    """
    if not 0.0 <= coefficient < 1.0:
        raise FeatureError("coefficient must be in [0, 1)")
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr.copy()
    out = np.empty_like(arr)
    out[0] = arr[0]
    if arr.size > 1:
        out[1:] = arr[1:] - coefficient * arr[:-1]
    return out.astype(np.float32)


__all__ = ["pre_emphasize"]
