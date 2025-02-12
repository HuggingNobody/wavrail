"""Tests for :func:`wavrail.audio.resample`."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.audio.resample import resample
from wavrail.errors import AudioError


def test_resample_identity_preserves_signal() -> None:
    samples = np.linspace(-1, 1, 32, dtype=np.float32)
    out = resample(samples, source_rate=16000, target_rate=16000)
    np.testing.assert_array_equal(out, samples.astype(np.float32))


def test_resample_upsamples_to_expected_length() -> None:
    samples = np.linspace(-1, 1, 100, dtype=np.float32)
    out = resample(samples, source_rate=16000, target_rate=48000)
    assert out.shape == (300,)
    assert out.dtype == np.float32


def test_resample_downsamples_to_expected_length() -> None:
    samples = np.linspace(-1, 1, 480, dtype=np.float32)
    out = resample(samples, source_rate=48000, target_rate=16000)
    assert out.shape == (160,)


def test_resample_rejects_zero_rates() -> None:
    samples = np.zeros(8, dtype=np.float32)
    with pytest.raises(AudioError):
        resample(samples, source_rate=0, target_rate=16000)
    with pytest.raises(AudioError):
        resample(samples, source_rate=16000, target_rate=0)


def test_resample_rejects_multi_channel() -> None:
    samples = np.zeros((8, 2), dtype=np.float32)
    with pytest.raises(AudioError):
        resample(samples, source_rate=16000, target_rate=16000)


def test_resample_empty_input_returns_empty() -> None:
    out = resample(np.zeros(0, dtype=np.float32), source_rate=16000, target_rate=8000)
    assert out.shape == (0,)
