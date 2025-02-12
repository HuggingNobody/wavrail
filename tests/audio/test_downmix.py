"""Tests for :func:`wavrail.audio.downmix_to_mono`."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.audio.downmix import downmix_to_mono
from wavrail.errors import AudioError


def test_downmix_stereo_returns_mono_float32() -> None:
    stereo = np.array([[0.1, 0.3], [0.2, 0.4], [0.0, -0.5]], dtype=np.float32)
    mono = downmix_to_mono(stereo)
    assert mono.shape == (3,)
    assert mono.dtype == np.float32
    np.testing.assert_allclose(mono, np.array([0.2, 0.3, -0.25], dtype=np.float32))


def test_downmix_mono_returns_copy() -> None:
    mono = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    result = downmix_to_mono(mono)
    assert result.shape == mono.shape
    assert not np.shares_memory(result, mono)
    np.testing.assert_array_equal(result, mono)


def test_downmix_rejects_high_dimensional_input() -> None:
    with pytest.raises(AudioError):
        downmix_to_mono(np.zeros((2, 2, 2), dtype=np.float32))


def test_downmix_rejects_zero_channel_audio() -> None:
    with pytest.raises(AudioError):
        downmix_to_mono(np.zeros((4, 0), dtype=np.float32))
