"""Frame blocking utilities.

The :func:`frame_signal` function slices a 1-D signal into a 2-D
matrix of overlapping frames, optionally padding the input so the
first and last frames are centred on real samples.
"""

from __future__ import annotations

import numpy as np

from wavrail.errors import FeatureError
from wavrail.types import AudioArray, FeatureArray, FrameConfig


def frame_signal(samples: AudioArray, config: FrameConfig) -> FeatureArray:
    """Block ``samples`` into overlapping frames.

    Returns a ``float32`` array of shape ``(num_frames, frame_length)``.
    When ``center`` is ``True`` the input is reflected-padded so the
    first frame is centred at sample ``0``; otherwise the first frame
    starts at sample ``0``.
    """
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise FeatureError("cannot frame an empty signal")
    frame_length = config.frame_length
    hop_length = config.hop_length
    if config.center:
        pad = frame_length // 2
        arr = _pad(arr, pad, config.pad_mode)
    num_frames = max(1, (arr.size - frame_length) // hop_length + 1)
    indices = np.arange(frame_length)[None, :] + hop_length * np.arange(num_frames)[:, None]
    return arr[indices].astype(np.float32)


def _pad(samples: AudioArray, pad: int, mode: str) -> AudioArray:
    if pad <= 0:
        return samples
    if mode == "constant":
        return np.pad(samples, (pad, pad), mode="constant")
    if mode == "edge":
        return np.pad(samples, (pad, pad), mode="edge")
    if mode == "reflect":
        return np.pad(samples, (pad, pad), mode="reflect")
    raise FeatureError(f"unknown pad_mode: {mode}")


def num_frames(num_samples: int, config: FrameConfig) -> int:
    """Return the number of frames produced for a given input length."""
    if num_samples <= 0:
        return 0
    length = num_samples + (config.frame_length // 2 if config.center else 0) * 2
    return max(1, (length - config.frame_length) // config.hop_length + 1)


__all__ = ["frame_signal", "num_frames"]
