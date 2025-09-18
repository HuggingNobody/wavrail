"""Spectral descriptors computed from power spectra.

Every function here consumes a magnitude-squared spectrum of shape
``(num_frames, num_bins)`` — the output of
:func:`wavrail.features.power_spectrum` — and returns one value per
frame as a ``float32`` array of shape ``(num_frames,)``. The
frequency axis is derived from ``sample_rate`` under the assumption
that ``num_bins == fft_size // 2 + 1``.

Frames with zero total energy have no spectral shape; centroid,
bandwidth, rolloff, and flatness are all defined to return ``0.0``
for them instead of dividing by zero. This convention is pinned by
the contract tests.
"""

from __future__ import annotations

import numpy as np

from wavrail.errors import FeatureError
from wavrail.types import FeatureArray


def _validate(spectra: FeatureArray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    if sample_rate <= 0:
        raise FeatureError("sample_rate must be positive")
    arr = np.asarray(spectra, dtype=np.float64)
    if arr.ndim != 2:
        raise FeatureError("spectra must be a 2-D array of shape (num_frames, num_bins)")
    if arr.shape[1] < 2:
        raise FeatureError("spectra must have at least two bins")
    if np.any(arr < 0.0):
        raise FeatureError("power spectra must be non-negative")
    n_fft = (arr.shape[1] - 1) * 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return arr, freqs


def spectral_centroid(spectra: FeatureArray, sample_rate: int) -> FeatureArray:
    """Return the energy-weighted mean frequency of each frame, in Hz."""
    arr, freqs = _validate(spectra, sample_rate)
    total = arr.sum(axis=1)
    weighted = arr @ freqs
    out = np.zeros(arr.shape[0], dtype=np.float64)
    nonzero = total > 0.0
    out[nonzero] = weighted[nonzero] / total[nonzero]
    return out.astype(np.float32)


def spectral_bandwidth(spectra: FeatureArray, sample_rate: int) -> FeatureArray:
    """Return the energy-weighted standard deviation around the centroid, in Hz."""
    arr, freqs = _validate(spectra, sample_rate)
    centroid = spectral_centroid(spectra, sample_rate).astype(np.float64)
    total = arr.sum(axis=1)
    deviation = freqs[None, :] - centroid[:, None]
    variance = (arr * deviation * deviation).sum(axis=1)
    out = np.zeros(arr.shape[0], dtype=np.float64)
    nonzero = total > 0.0
    out[nonzero] = np.sqrt(variance[nonzero] / total[nonzero])
    return out.astype(np.float32)


def spectral_rolloff(
    spectra: FeatureArray,
    sample_rate: int,
    percent: float = 0.85,
) -> FeatureArray:
    """Return the frequency below which ``percent`` of the energy sits.

    The rolloff of a zero-energy frame is ``0.0``. ``percent`` must
    be in ``(0, 1]``.
    """
    if not 0.0 < percent <= 1.0:
        raise FeatureError("percent must be in (0, 1]")
    arr, freqs = _validate(spectra, sample_rate)
    total = arr.sum(axis=1)
    cumulative = np.cumsum(arr, axis=1)
    out = np.zeros(arr.shape[0], dtype=np.float64)
    for row in range(arr.shape[0]):
        if total[row] <= 0.0:
            continue
        idx = int(np.searchsorted(cumulative[row], percent * total[row] - 1e-12))
        idx = min(idx, arr.shape[1] - 1)
        out[row] = freqs[idx]
    return out.astype(np.float32)


def spectral_flatness(
    spectra: FeatureArray,
    sample_rate: int,
    eps: float = 1e-10,
) -> FeatureArray:
    """Return the geometric-to-arithmetic mean ratio of each frame's spectrum.

    Values near ``1.0`` indicate noise-like flat spectra; values near
    ``0.0`` indicate tonal content. Zero-energy frames return ``0.0``
    and the result is clamped to ``[0, 1]``. The ``eps`` floor keeps
    the geometric mean defined when individual bins are exactly zero.
    ``sample_rate`` is accepted for a uniform signature even though
    the ratio itself is scale-free.
    """
    arr, _ = _validate(spectra, sample_rate)
    if eps <= 0.0:
        raise FeatureError("eps must be positive")
    total = arr.sum(axis=1)
    floored = np.maximum(arr, eps)
    geometric = np.exp(np.log(floored).mean(axis=1))
    arithmetic = arr.mean(axis=1)
    out = np.zeros(arr.shape[0], dtype=np.float64)
    nonzero = total > 0.0
    out[nonzero] = geometric[nonzero] / (arithmetic[nonzero] + eps)
    return np.minimum(out, 1.0).astype(np.float32)


__all__ = [
    "spectral_bandwidth",
    "spectral_centroid",
    "spectral_flatness",
    "spectral_rolloff",
]
