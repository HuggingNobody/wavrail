"""Time-domain frame energy helpers.

:func:`frame_energy` sums the squared samples of each frame, and
:func:`log_frame_energy` applies a floored natural logarithm so
silent frames stay finite. Both consume frames of shape
``(num_frames, frame_length)`` and return ``float32`` arrays of
shape ``(num_frames,)``.

The log floor is a documented contract: energies below ``floor``
are clamped before taking the logarithm, so the smallest
representable output is ``log(floor)``.
"""

from __future__ import annotations

import numpy as np

from wavrail.errors import FeatureError
from wavrail.types import FeatureArray


def _validate_frames(frames: FeatureArray) -> np.ndarray:
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 2:
        raise FeatureError("frames must be a 2-D array of shape (num_frames, frame_length)")
    return arr


def frame_energy(frames: FeatureArray) -> FeatureArray:
    """Return the sum of squared samples for every frame."""
    arr = _validate_frames(frames).astype(np.float64)
    return (arr * arr).sum(axis=1).astype(np.float32)


def log_frame_energy(frames: FeatureArray, floor: float = 1e-10) -> FeatureArray:
    """Return ``log(max(energy, floor))`` for every frame (natural log)."""
    if floor <= 0.0:
        raise FeatureError("floor must be positive")
    energy = frame_energy(frames).astype(np.float64)
    return np.log(np.maximum(energy, floor)).astype(np.float32)


def zero_crossing_rate(frames: FeatureArray) -> FeatureArray:
    """Return the sign-change rate of every frame, in ``[0, 1]``.

    A crossing is counted whenever consecutive samples flip sign bit
    (``np.signbit``), so an exact zero is treated as positive. The
    count is divided by ``frame_length - 1``; frames shorter than two
    samples are rejected because the rate is undefined there.
    High rates indicate noise-like or unvoiced frames, low rates
    indicate voiced or tonal frames.
    """
    arr = _validate_frames(frames)
    if arr.shape[1] < 2:
        raise FeatureError("frames must contain at least two samples")
    signs = np.signbit(arr.astype(np.float32))
    crossings = np.count_nonzero(signs[:, 1:] != signs[:, :-1], axis=1)
    return (crossings / (arr.shape[1] - 1)).astype(np.float32)


__all__ = ["frame_energy", "log_frame_energy", "zero_crossing_rate"]
