"""Common data classes and type aliases used throughout the wavrail package.

These types are intentionally small and dependency-free so that every
subpackage can import them without circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

#: A 1-D float32 array of audio samples in the range ``[-1, 1]``.
AudioArray: TypeAlias = np.ndarray

#: A 2-D float array of shape ``(num_frames, num_features)``.
FeatureArray: TypeAlias = np.ndarray

#: A 2-D float array of shape ``(num_frames, num_classes)`` of posterior
#: probabilities or log-probabilities used for CTC decoding.
PosteriorArray: TypeAlias = np.ndarray


@dataclass(frozen=True)
class AudioMeta:
    """Static metadata for an audio buffer."""

    sample_rate: int
    num_channels: int
    sample_width: int
    duration_s: float
    num_samples: int

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if self.sample_width not in {2, 3, 4}:
            raise ValueError("sample_width must be 2, 3, or 4 bytes")
        if self.num_samples < 0:
            raise ValueError("num_samples must be non-negative")


@dataclass(frozen=True)
class FrameConfig:
    """Configuration for blocking a signal into overlapping frames."""

    frame_length: int
    hop_length: int
    pad_mode: str = "constant"
    center: bool = True

    def __post_init__(self) -> None:
        if self.frame_length <= 0:
            raise ValueError("frame_length must be positive")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.frame_length < self.hop_length:
            raise ValueError("frame_length must be >= hop_length")
        if self.pad_mode not in {"constant", "edge", "reflect"}:
            raise ValueError(f"unknown pad_mode: {self.pad_mode}")


@dataclass(frozen=True)
class MelConfig:
    """Configuration for mel filterbank construction."""

    sample_rate: int
    num_mels: int
    fft_size: int
    fmin: float = 0.0
    fmax: float | None = None
    scale: str = "htk"
    log_floor: float = 1e-10

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.num_mels <= 0:
            raise ValueError("num_mels must be positive")
        if self.fft_size <= 0 or self.fft_size % 2 == 1:
            raise ValueError("fft_size must be a positive even integer")
        if self.fmin < 0:
            raise ValueError("fmin must be non-negative")
        if self.fmax is not None and self.fmax <= self.fmin:
            raise ValueError("fmax must be > fmin when provided")
        if self.scale not in {"htk", "slaney"}:
            raise ValueError(f"unknown mel scale convention: {self.scale}")
        if self.log_floor <= 0:
            raise ValueError("log_floor must be positive")
        if self.fmax is None:
            object.__setattr__(self, "fmax", self.sample_rate / 2)


__all__ = [
    "AudioArray",
    "AudioMeta",
    "FeatureArray",
    "FrameConfig",
    "MelConfig",
    "PosteriorArray",
]
