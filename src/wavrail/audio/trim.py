"""Head/tail silence trimming.

:func:`trim_silence` removes leading and trailing samples whose
absolute amplitude does not exceed a dBFS threshold. The interior of
the signal is never touched, even when it contains quiet passages —
use :func:`wavrail.audio.detect_silence_regions` when silence *inside*
an utterance matters.

Contracts pinned by tests: a fully quiet buffer trims to an empty
array, samples exactly at the threshold count as quiet, and the input
buffer is never mutated.
"""

from __future__ import annotations

import numpy as np

from wavrail.audio.energy import from_dbfs
from wavrail.errors import AudioError
from wavrail.types import AudioArray


def trim_silence(samples: AudioArray, threshold_db: float = -60.0) -> AudioArray:
    """Return a copy of ``samples`` without quiet head and tail."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim != 1:
        raise AudioError("samples must be a 1-D array")
    limit = from_dbfs(threshold_db)  # validates threshold <= 0 dBFS
    loud = np.abs(arr) > limit
    if not np.any(loud):
        return np.zeros(0, dtype=np.float32)
    first = int(np.argmax(loud))
    last = int(arr.size - np.argmax(loud[::-1]))
    return arr[first:last].copy()


__all__ = ["trim_silence"]
