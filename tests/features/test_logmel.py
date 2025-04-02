"""Tests for :func:`wavrail.features.log_mel_spectrogram`."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.errors import FeatureError
from wavrail.features.logmel import log_mel_spectrogram
from wavrail.types import FrameConfig, MelConfig


def _log_mel(samples: np.ndarray, sr: int = 16000) -> np.ndarray:
    return log_mel_spectrogram(
        samples,
        sample_rate=sr,
        frame_config=FrameConfig(frame_length=400, hop_length=160, center=True),
        mel_config=MelConfig(sample_rate=sr, num_mels=40, fft_size=512),
    )


def test_log_mel_shape() -> None:
    samples = np.random.default_rng(0).normal(-0.1, 0.1, size=16000).astype(np.float32)
    log_mel = _log_mel(samples)
    assert log_mel.ndim == 2
    assert log_mel.shape[1] == 40
    assert log_mel.shape[0] > 0
    assert log_mel.dtype == np.float32


def test_log_mel_values_below_log_floor() -> None:
    samples = np.zeros(8000, dtype=np.float32)
    log_mel = log_mel_spectrogram(
        samples,
        sample_rate=16000,
        frame_config=FrameConfig(frame_length=400, hop_length=160),
        mel_config=MelConfig(sample_rate=16000, num_mels=40, fft_size=512, log_floor=1e-3),
    )
    assert np.all(log_mel <= np.log(1e-3) + 1e-5)


def test_log_mel_rejects_sample_rate_mismatch() -> None:
    samples = np.zeros(1000, dtype=np.float32)
    with pytest.raises(FeatureError):
        log_mel_spectrogram(
            samples,
            sample_rate=16000,
            frame_config=FrameConfig(frame_length=400, hop_length=160),
            mel_config=MelConfig(sample_rate=8000, num_mels=40, fft_size=512),
        )


def test_log_mel_rejects_fft_smaller_than_frame() -> None:
    samples = np.zeros(1000, dtype=np.float32)
    with pytest.raises(FeatureError):
        log_mel_spectrogram(
            samples,
            sample_rate=16000,
            frame_config=FrameConfig(frame_length=400, hop_length=160),
            mel_config=MelConfig(sample_rate=16000, num_mels=40, fft_size=256),
        )
