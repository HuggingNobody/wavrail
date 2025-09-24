"""VAD streaming equivalence and hangover state handling."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.audio import synthesize_pulse
from wavrail.errors import FeatureError
from wavrail.features import EnergyVad, detect_voice_activity, frame_signal
from wavrail.types import FrameConfig


def _frames() -> np.ndarray:
    samples = synthesize_pulse(0.02, 0.04, 0.2, 16000)
    return frame_signal(samples, FrameConfig(frame_length=400, hop_length=160, center=False))


@pytest.mark.parametrize("hangover", [0, 1, 2, 5])
def test_streaming_matches_batch(hangover):
    frames = _frames()
    vad = EnergyVad(hangover_frames=hangover)
    streamed = [vad.process_frame(row) for row in frames]
    assert streamed == list(detect_voice_activity(frames, hangover_frames=hangover))


def test_reset_clears_pending_hangover():
    vad = EnergyVad(threshold_db=-40.0, hangover_frames=3)
    assert vad.process_frame(np.ones(80, dtype=np.float32))
    vad.reset()
    assert not vad.process_frame(np.zeros(80, dtype=np.float32))


def test_hangover_extends_exactly_n_frames():
    frames = np.vstack([np.ones((1, 64)), np.zeros((5, 64))]).astype(np.float32)
    flags = detect_voice_activity(frames, hangover_frames=2)
    assert list(flags) == [True, True, True, False, False, False]


def test_validation():
    with pytest.raises(FeatureError):
        EnergyVad(hangover_frames=-1)
    with pytest.raises(FeatureError):
        detect_voice_activity(np.zeros(64, dtype=np.float32))
    vad = EnergyVad()
    with pytest.raises(FeatureError):
        vad.process_frame(np.zeros(0, dtype=np.float32))
