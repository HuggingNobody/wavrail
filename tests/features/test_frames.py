"""Tests for :func:`wavrail.features.frame_signal`."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.errors import FeatureError
from wavrail.features.frames import frame_signal, num_frames
from wavrail.types import FrameConfig


def test_frame_count_matches_formula() -> None:
    config = FrameConfig(frame_length=4, hop_length=2, center=False)
    assert num_frames(10, config) == 4
    assert num_frames(4, config) == 1
    assert num_frames(2, config) == 0


def test_centered_framing_includes_padding() -> None:
    config = FrameConfig(frame_length=4, hop_length=2, center=True)
    samples = np.arange(10, dtype=np.float32)
    frames = frame_signal(samples, config)
    expected_count = (10 + 2 + 2 - 4) // 2 + 1
    assert frames.shape == (expected_count, 4)
    assert frames.dtype == np.float32


def test_left_aligned_framing() -> None:
    config = FrameConfig(frame_length=4, hop_length=2, center=False)
    samples = np.arange(10, dtype=np.float32)
    frames = frame_signal(samples, config)
    assert frames.shape == (4, 4)
    np.testing.assert_array_equal(frames[0], samples[:4])


def test_empty_signal_raises() -> None:
    config = FrameConfig(frame_length=4, hop_length=2)
    with pytest.raises(FeatureError):
        frame_signal(np.zeros(0, dtype=np.float32), config)


def test_reflect_pad_differs_from_constant() -> None:
    config_const = FrameConfig(frame_length=8, hop_length=4, center=True, pad_mode="constant")
    config_reflect = FrameConfig(frame_length=8, hop_length=4, center=True, pad_mode="reflect")
    samples = np.arange(16, dtype=np.float32)
    const_frames = frame_signal(samples, config_const)
    reflect_frames = frame_signal(samples, config_reflect)
    assert not np.array_equal(const_frames[0], reflect_frames[0])
