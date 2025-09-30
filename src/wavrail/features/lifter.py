"""Cepstral liftering weights.

Liftering (a deliberate pun on "filtering") reweights cepstral
coefficients to emphasise the mid-range coefficients that carry most
of the speech-relevant information. The weights follow the HTK
convention: coefficient ``k >= 1`` is scaled by
``1 + (lifter / 2) * sin(pi * k / lifter)`` and coefficient 0 is left
untouched. ``lifter = 0`` disables liftering and yields all-ones
weights.

The sine formula is applied for every ``k >= 1``; for ``k > lifter``
the weights oscillate back down. Typical MFCC configurations keep
``num_coeffs`` well below ``lifter`` so weights stay in ``[1, lifter]``.
"""

from __future__ import annotations

import numpy as np

from wavrail.errors import FeatureError
from wavrail.types import FeatureArray


def cepstral_lifter(num_coeffs: int, lifter: int = 22) -> np.ndarray:
    """Return the ``(num_coeffs,)`` lifter weight vector."""
    if num_coeffs < 1:
        raise FeatureError("num_coeffs must be positive")
    if lifter < 0:
        raise FeatureError("lifter must be non-negative")
    weights = np.ones(num_coeffs, dtype=np.float64)
    if lifter > 0:
        k = np.arange(1, num_coeffs, dtype=np.float64)
        weights[1:] = 1.0 + (lifter / 2.0) * np.sin(np.pi * k / lifter)
    return weights


def apply_lifter(coeffs: FeatureArray, lifter: int = 22) -> FeatureArray:
    """Scale every cepstral row of ``coeffs`` by the lifter weights."""
    arr = np.asarray(coeffs, dtype=np.float64)
    if arr.ndim != 2:
        raise FeatureError("coeffs must be a 2-D array of shape (num_frames, num_coeffs)")
    weights = cepstral_lifter(arr.shape[1], lifter)
    return (arr * weights).astype(np.float32)


__all__ = ["apply_lifter", "cepstral_lifter"]
