"""Tapering window functions.

Each function returns a ``float32`` numpy array of length
``frame_length``. The ``symmetric`` flag controls whether the
window is symmetric (``True``, periodic-DTFT convention) or
periodic (``False``, useful for overlap-add reconstruction).
"""

from __future__ import annotations

import numpy as np

from wavrail.errors import FeatureError


def hann_window(frame_length: int, symmetric: bool = True) -> np.ndarray:
    """Hann (raised-cosine) window."""
    return _general_hamming(frame_length, alpha=0.5, symmetric=symmetric)


def hamming_window(frame_length: int, symmetric: bool = True) -> np.ndarray:
    """Hamming window with alpha=0.54."""
    return _general_hamming(frame_length, alpha=0.46, symmetric=symmetric)


def blackman_window(frame_length: int, symmetric: bool = True) -> np.ndarray:
    """Three-term Blackman window."""
    if frame_length < 3:
        raise FeatureError("blackman window requires length >= 3")
    n = np.arange(frame_length, dtype=np.float64)
    denom = (frame_length - 1) if frame_length > 1 else 1
    if not symmetric:
        denom = frame_length
    n = n / denom
    w = 0.42 - 0.5 * np.cos(2.0 * np.pi * n) + 0.08 * np.cos(4.0 * np.pi * n)
    return w.astype(np.float32)


def rectangular_window(frame_length: int) -> np.ndarray:
    """All-ones window."""
    if frame_length <= 0:
        raise FeatureError("window length must be positive")
    return np.ones(frame_length, dtype=np.float32)


def _general_hamming(frame_length: int, alpha: float, symmetric: bool) -> np.ndarray:
    if frame_length <= 0:
        raise FeatureError("window length must be positive")
    n = np.arange(frame_length, dtype=np.float64)
    denom = (frame_length - 1) if frame_length > 1 else 1
    if not symmetric:
        denom = frame_length
    n = n / denom
    w = alpha - (1.0 - alpha) * np.cos(2.0 * np.pi * n)
    return w.astype(np.float32)


__all__ = ["blackman_window", "hamming_window", "hann_window", "rectangular_window"]
