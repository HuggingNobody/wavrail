"""Tests for the :mod:`wavrail.audio.quality` helpers."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.audio.quality import clipping_detected, silence_ratio
from wavrail.errors import AudioError


def test_clipping_detected_basic() -> None:
    assert not clipping_detected(np.zeros(8, dtype=np.float32))
    assert clipping_detected(np.array([0.5, 1.0, -0.5], dtype=np.float32))
    assert clipping_detected(np.array([0.99, -1.0, 0.5], dtype=np.float32), threshold=0.99)


def test_clipping_detected_multi_channel() -> None:
    samples = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    assert clipping_detected(samples)


def test_clipping_detected_empty_buffer() -> None:
    assert not clipping_detected(np.zeros(0, dtype=np.float32))


def test_clipping_detected_rejects_invalid_threshold() -> None:
    with pytest.raises(AudioError):
        clipping_detected(np.zeros(1, dtype=np.float32), threshold=0.0)
    with pytest.raises(AudioError):
        clipping_detected(np.zeros(1, dtype=np.float32), threshold=1.5)


def test_silence_ratio_default_threshold() -> None:
    samples = np.array([0.0, 0.0005, 0.5, -0.5], dtype=np.float32)
    ratio = silence_ratio(samples)
    assert ratio == pytest.approx(0.5)


def test_silence_ratio_multi_channel() -> None:
    samples = np.array([[0.0, 0.0], [0.0, 0.5]], dtype=np.float32)
    assert silence_ratio(samples) == pytest.approx(0.75)


def test_silence_ratio_empty_buffer() -> None:
    assert silence_ratio(np.zeros(0, dtype=np.float32)) == 0.0


def test_silence_ratio_rejects_negative_threshold() -> None:
    with pytest.raises(AudioError):
        silence_ratio(np.zeros(1, dtype=np.float32), threshold=-0.1)
