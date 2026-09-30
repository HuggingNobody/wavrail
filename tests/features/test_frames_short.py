"""Edge cases for frame blocking."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.errors import FeatureError
from wavrail.features.frames import frame_signal, num_frames
from wavrail.types import FrameConfig


def test_short_signal_center_padding() -> None:
    config = FrameConfig(frame_length=10, hop_length=5, center=True)
    frames = frame_signal(np.ones(5, dtype=np.float32), config)
    assert frames.shape[1] == 10
    assert frames.shape[0] >= 1


def test_num_frames_matches_blocking() -> None:
    config = FrameConfig(frame_length=8, hop_length=4)
    frames = frame_signal(np.ones(32, dtype=np.float32), config)
    assert frames.shape[0] == num_frames(32, config)


def test_rejects_empty_signal() -> None:
    config = FrameConfig(frame_length=8, hop_length=4)
    with pytest.raises(FeatureError):
        frame_signal(np.zeros(0, dtype=np.float32), config)


def test_pad_mode_constant() -> None:
    config = FrameConfig(frame_length=8, hop_length=4, center=True, pad_mode="constant")
    frames = frame_signal(np.ones(16, dtype=np.float32), config)
    assert frames.shape[0] >= 1


def test_pad_mode_edge() -> None:
    config = FrameConfig(frame_length=8, hop_length=4, center=True, pad_mode="edge")
    frames = frame_signal(np.ones(16, dtype=np.float32), config)
    assert frames.shape[0] >= 1


def test_rejects_unknown_pad_mode() -> None:
    config = FrameConfig(frame_length=8, hop_length=4, center=True, pad_mode="bogus")
    with pytest.raises(FeatureError, match="pad_mode"):
        frame_signal(np.ones(16, dtype=np.float32), config)


def test_short_signal_without_center_returns_empty_frames() -> None:
    config = FrameConfig(frame_length=8, hop_length=4, center=False)
    frames = frame_signal(np.ones(5, dtype=np.float32), config)

    assert frames.shape == (0, 8)
    assert frames.dtype == np.float32
    assert frames.shape[0] == num_frames(5, config)
