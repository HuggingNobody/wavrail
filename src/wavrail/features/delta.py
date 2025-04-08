"""Delta and delta-delta derivative features.

:func:`compute_deltas` implements the standard finite-difference
operator with a configurable context window ``N`` (HTK default
``N = 2``). The function pads the input by reflection so the
output has the same number of frames as the input.
"""

from __future__ import annotations

import numpy as np

from wavrail.errors import FeatureError
from wavrail.types import FeatureArray


def compute_deltas(features: FeatureArray, window: int = 2) -> FeatureArray:
    """Compute first-order delta features with a symmetric context window."""
    if window <= 0:
        raise FeatureError("window must be positive")
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2:
        raise FeatureError("features must be a 2-D array")
    if arr.shape[0] < 2:
        raise FeatureError("need at least 2 frames to compute deltas")
    padded = np.pad(arr, ((window, window), (0, 0)), mode="edge")
    denominator = 2.0 * sum(n * n for n in range(1, window + 1))
    deltas = np.zeros_like(arr)
    for n in range(1, window + 1):
        future = padded[window + n : window + n + arr.shape[0]]
        past = padded[window - n : window - n + arr.shape[0]]
        deltas += n * (future - past)
    return (deltas / denominator).astype(np.float32)


__all__ = ["compute_deltas"]
