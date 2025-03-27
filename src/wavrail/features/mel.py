"""Mel filterbank construction.

Two mel scale conventions are supported. The HTK convention uses
``2595 * log10(1 + f / 700)``. The Slaney convention uses a
piecewise-linear approximation that is faster but slightly less
accurate for very high frequencies. Triangular filters are built
between equally-spaced mel points.
"""

from __future__ import annotations

import numpy as np

from wavrail.errors import FeatureError
from wavrail.types import MelConfig


def hz_to_mel(hz: np.ndarray, scale: str) -> np.ndarray:
    """Convert Hz to mel scale."""
    if scale == "htk":
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    if scale == "slaney":
        f_sp = 1000.0
        min_log_hz = 15.0
        return np.where(
            hz >= f_sp,
            np.log(hz / min_log_hz) * 27.0 / np.log(6.4),
            hz / f_sp * 200.0,
        )
    raise FeatureError(f"unknown mel scale: {scale}")


def mel_to_hz(mel: np.ndarray, scale: str) -> np.ndarray:
    """Convert mel scale to Hz."""
    if scale == "htk":
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    if scale == "slaney":
        f_sp = 1000.0
        min_log_hz = 15.0
        return np.where(
            mel >= 15.0,
            np.exp((mel / 27.0) * np.log(6.4)) * min_log_hz,
            f_sp / 200.0 * mel,
        )
    raise FeatureError(f"unknown mel scale: {scale}")


def mel_filterbank(config: MelConfig) -> np.ndarray:
    """Construct a triangular mel filterbank matrix.

    Returns a ``float32`` array of shape
    ``(num_mels, config.fft_size // 2 + 1)``. Each row sums to 1.0
    so that downstream log-mel normalisation is straightforward.
    """
    if config.fft_size <= 0:
        raise FeatureError("fft_size must be positive")
    if config.num_mels <= 0:
        raise FeatureError("num_mels must be positive")
    num_bins = config.fft_size // 2 + 1
    mel_low = hz_to_mel(np.array([config.fmin], dtype=np.float64), config.scale)[0]
    mel_high = hz_to_mel(np.array([config.fmax], dtype=np.float64), config.scale)[0]
    mel_points = np.linspace(mel_low, mel_high, config.num_mels + 2, dtype=np.float64)
    hz_centers = mel_to_hz(mel_points, config.scale)
    bin_indices = np.floor((config.fft_size + 1) * hz_centers / config.sample_rate).astype(np.int64)
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    fb = np.zeros((config.num_mels, num_bins), dtype=np.float32)
    for m in range(config.num_mels):
        left = int(bin_indices[m])
        center = int(bin_indices[m + 1])
        right = int(bin_indices[m + 2])
        if center > left:
            denom = float(hz_centers[m + 1] - hz_centers[m])
            if denom <= 0:
                denom = 1e-12
            fb[m, left:center] = (hz_centers[m + 1] - hz_centers[m]) / denom
        if right > center:
            denom = float(hz_centers[m + 2] - hz_centers[m + 1])
            if denom <= 0:
                denom = 1e-12
            fb[m, center:right] = (hz_centers[m + 2] - hz_centers[m + 1]) / denom
    return fb


__all__ = ["hz_to_mel", "mel_filterbank", "mel_to_hz"]
