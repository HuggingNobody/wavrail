"""Channel downmix utilities.

The default policy averages all input channels with equal weight.
This is the simplest downmix that preserves loudness for
non-correlated channels. Production ASR systems often prefer a
Loudness Units Full Scale weighted downmix (ITU-R BS.1770) but
the equal-weight policy is sufficient for wavrail's synthetic
fixtures and for the streaming pipeline.
"""

from __future__ import annotations

import numpy as np

from wavrail.errors import AudioError
from wavrail.types import AudioArray


def downmix_to_mono(samples: AudioArray) -> AudioArray:
    """Average all channels into a single mono ``float32`` array.

    Mono inputs are returned as ``float32`` copies.
    """
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 1:
        return arr.copy()
    if arr.ndim == 2:
        if arr.shape[1] == 0:
            raise AudioError("cannot downmix zero-channel audio")
        return arr.mean(axis=1).astype(np.float32)
    raise AudioError("samples must be 1-D or 2-D")


__all__ = ["downmix_to_mono"]
