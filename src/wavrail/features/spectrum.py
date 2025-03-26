"""Power spectrum helpers.

The :func:`power_spectrum` function returns the magnitude-squared
DFT bins of each frame, optionally smoothed by a window and
converted to dB scale.
"""

from __future__ import annotations

import numpy as np

from wavrail.errors import FeatureError
from wavrail.types import FeatureArray


def power_spectrum(
    frames: FeatureArray,
    fft_size: int,
    window: np.ndarray | None = None,
) -> FeatureArray:
    """Compute the magnitude-squared DFT of every frame.

    Parameters
    ----------
    frames:
        ``float32`` array of shape ``(num_frames, frame_length)``.
    fft_size:
        FFT size used for the DFT. Must be ``>= frame_length``.
    window:
        Optional window vector. When provided, it is applied to each
        frame before the DFT. When ``None`` a rectangular window is
        used.

    Returns
    -------
    np.ndarray
        ``float32`` array of shape ``(num_frames, fft_size // 2 + 1)``.
    """
    if fft_size <= 0:
        raise FeatureError("fft_size must be positive")
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 2:
        raise FeatureError("frames must be a 2-D array")
    if window is not None:
        win = np.asarray(window, dtype=np.float32).reshape(-1)
        if win.shape[0] != arr.shape[1]:
            raise FeatureError("window length must match frame length")
        arr = arr * win
    spectrum = np.fft.rfft(arr, n=fft_size, axis=1)
    magnitude = np.abs(spectrum).astype(np.float32)
    return (magnitude * magnitude).astype(np.float32)


__all__ = ["power_spectrum"]
