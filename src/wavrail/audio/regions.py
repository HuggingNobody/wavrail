"""Silent-run detection over a sample buffer.

:func:`detect_silence_regions` returns half-open ``(start, end)``
sample-index pairs for every run of consecutive quiet samples. A
sample is quiet when ``abs(sample) <= from_dbfs(threshold_db)``; runs
shorter than ``min_region_samples`` are dropped, which keeps single
zero crossings of a loud sine wave from polluting the result.

The detector is sample-exact rather than frame-based: region borders
land on the first/last quiet sample of a run, not on frame hops.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from wavrail.audio.energy import from_dbfs
from wavrail.errors import AudioError
from wavrail.types import AudioArray


def detect_silence_regions(
    samples: AudioArray,
    threshold_db: float = -60.0,
    min_region_samples: int = 1,
) -> list[tuple[int, int]]:
    """Return half-open sample ranges of quiet runs at or below ``threshold_db``."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim != 1:
        raise AudioError("samples must be a 1-D array")
    if min_region_samples < 1:
        raise AudioError("min_region_samples must be positive")
    limit = from_dbfs(threshold_db)
    quiet = np.abs(arr) <= limit
    if not np.any(quiet):
        return []
    changes = np.flatnonzero(quiet[1:] != quiet[:-1]) + 1
    bounds = np.concatenate([[0], changes, [arr.size]])
    regions: list[tuple[int, int]] = []
    for start, end in pairwise(bounds):
        if quiet[start] and int(end - start) >= min_region_samples:
            regions.append((int(start), int(end)))
    return regions


__all__ = ["detect_silence_regions"]
