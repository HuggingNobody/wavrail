"""Quality heuristics for audio buffers.

The functions in this module answer simple yes/no questions about
an audio buffer. ``clipping_detected`` checks whether any sample
sits at the extremes of the representable range. ``silence_ratio``
returns the fraction of samples whose absolute amplitude falls
below a configurable threshold.

Both functions treat multi-channel arrays the same way as
mono arrays: they flatten to a 1-D view before counting. Callers
that need per-channel metrics can reshape the input first.
"""

from __future__ import annotations

import numpy as np

from wavrail.errors import AudioError
from wavrail.types import AudioArray


def clipping_detected(
    samples: AudioArray,
    threshold: float = 0.999,
) -> bool:
    """Return ``True`` if any sample exceeds ``threshold`` in absolute value."""
    if not 0.0 < threshold <= 1.0:
        raise AudioError("threshold must be in (0, 1]")
    flat = np.asarray(samples, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return False
    return bool(np.any(np.abs(flat) >= threshold))


def silence_ratio(
    samples: AudioArray,
    threshold: float = 1e-3,
) -> float:
    """Return the fraction of samples whose absolute amplitude is below ``threshold``."""
    if threshold < 0.0:
        raise AudioError("threshold must be non-negative")
    flat = np.asarray(samples, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return 0.0
    return float(np.mean(np.abs(flat) < threshold))


__all__ = ["clipping_detected", "silence_ratio"]
