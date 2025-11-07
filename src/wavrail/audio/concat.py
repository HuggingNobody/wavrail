"""Segment concatenation with optional linear crossfade.

:func:`concatenate` joins 1-D ``float32`` segments end to end. With
``crossfade_samples > 0`` the tail of the previous output and the head
of the next segment overlap over ``crossfade_samples`` samples using a
linear ramp from ``1`` to ``0`` (inclusive endpoints), which avoids
the discontinuity click of a hard splice at the cost of a short
amplitude dip for uncorrelated material.

Every result is a fresh array; inputs are never mutated, and list or
tuple inputs are accepted anywhere a NumPy array is.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from wavrail.errors import AudioError
from wavrail.types import AudioArray


def concatenate(
    segments: Sequence[AudioArray],
    crossfade_samples: int = 0,
) -> AudioArray:
    """Join ``segments``, overlapping neighbours by ``crossfade_samples``."""
    if len(segments) == 0:
        raise AudioError("segments must contain at least one segment")
    arrays = []
    for segment in segments:
        arr = np.asarray(segment, dtype=np.float32)
        if arr.ndim != 1:
            raise AudioError("every segment must be a 1-D array")
        arrays.append(arr)
    if crossfade_samples < 0:
        raise AudioError("crossfade_samples must be non-negative")
    if crossfade_samples > 0 and min(arr.size for arr in arrays) < crossfade_samples:
        raise AudioError("crossfade_samples exceeds the shortest segment")
    out = arrays[0].copy()
    if crossfade_samples == 0:
        for arr in arrays[1:]:
            out = np.concatenate([out, arr])
        return out.astype(np.float32)
    ramp = np.linspace(1.0, 0.0, crossfade_samples, dtype=np.float32)
    for arr in arrays[1:]:
        overlap = out[-crossfade_samples:] * ramp + arr[:crossfade_samples] * (1.0 - ramp)
        out = np.concatenate([out[:-crossfade_samples], overlap, arr[crossfade_samples:]])
    return out.astype(np.float32)


__all__ = ["concatenate"]
