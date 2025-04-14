"""Golden contract tests for the features subpackage.

Each test pins a documented shape contract so that future
refactors of the underlying primitives surface as a test failure
with a clear message.
"""

from __future__ import annotations

import numpy as np

from wavrail.features import (
    log_mel_spectrogram,
    mel_filterbank,
)
from wavrail.types import FrameConfig, MelConfig


def test_log_mel_shape_contract() -> None:
    samples = np.zeros(16000, dtype=np.float32)
    log_mel = log_mel_spectrogram(
        samples,
        sample_rate=16000,
        frame_config=FrameConfig(frame_length=400, hop_length=160, center=True),
        mel_config=MelConfig(sample_rate=16000, num_mels=80, fft_size=512),
    )
    expected_frames = (16000 + 200 + 200 - 400) // 160 + 1
    assert log_mel.shape == (expected_frames, 80)


def test_mel_filterbank_shape_contract() -> None:
    fb = mel_filterbank(MelConfig(sample_rate=16000, num_mels=64, fft_size=512))
    assert fb.shape == (64, 257)


def test_log_mel_is_finite_for_silence() -> None:
    samples = np.zeros(8000, dtype=np.float32)
    log_mel = log_mel_spectrogram(
        samples,
        sample_rate=16000,
        frame_config=FrameConfig(frame_length=400, hop_length=160, center=False),
        mel_config=MelConfig(sample_rate=16000, num_mels=40, fft_size=512),
    )
    assert np.isfinite(log_mel).all()
