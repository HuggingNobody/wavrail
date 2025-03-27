"""Tests for the mel filterbank module."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.features.mel import hz_to_mel, mel_filterbank, mel_to_hz
from wavrail.types import MelConfig


def test_filterbank_shape() -> None:
    config = MelConfig(sample_rate=16000, num_mels=80, fft_size=512)
    fb = mel_filterbank(config)
    assert fb.shape == (80, 257)
    assert fb.dtype == np.float32


def test_filterbank_rows_are_nonnegative() -> None:
    config = MelConfig(sample_rate=16000, num_mels=40, fft_size=256)
    fb = mel_filterbank(config)
    row_sums = fb.sum(axis=1)
    assert (row_sums >= 0.0).all()
    assert (row_sums > 0.0).all()


def test_hz_mel_roundtrip_htk() -> None:
    hz = np.array([0.0, 500.0, 1000.0, 4000.0, 8000.0])
    mel = hz_to_mel(hz, scale="htk")
    hz_back = mel_to_hz(mel, scale="htk")
    np.testing.assert_allclose(hz_back, hz, atol=1e-3)


def test_filterbank_rejects_invalid_config() -> None:
    with pytest.raises(ValueError):
        mel_filterbank(MelConfig(sample_rate=16000, num_mels=10, fft_size=256, fmin=-1.0))
    with pytest.raises(ValueError):
        mel_filterbank(MelConfig(sample_rate=16000, num_mels=10, fft_size=256, scale="bogus"))


def test_htk_and_slaney_disagree_at_high_frequencies() -> None:
    hz = np.array([8000.0])
    htk = hz_to_mel(hz, scale="htk")
    slaney = hz_to_mel(hz, scale="slaney")
    assert not np.allclose(htk, slaney, atol=1.0)
