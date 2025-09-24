"""Energy VAD flags on a pulse-in-silence fixture.

The fixture is 0.2 s at 16 kHz with an amplitude-1.0 pulse from
0.02 s to 0.06 s (samples 320..959). Frames are 400 samples with a
160-sample hop and no centering, so frame ``k`` covers samples
``[160k, 160k + 400)``: frames 0..5 overlap the pulse, frame 6 is
the first fully silent frame, and there are 18 frames in total.
"""

from __future__ import annotations

import numpy as np

from wavrail.audio import synthesize_pulse
from wavrail.features import detect_voice_activity, frame_signal
from wavrail.types import FrameConfig


def _frames() -> np.ndarray:
    samples = synthesize_pulse(0.02, 0.04, 0.2, 16000, amplitude=1.0)
    return frame_signal(samples, FrameConfig(frame_length=400, hop_length=160, center=False))


def test_pulse_region_is_voiced():
    frames = _frames()
    flags = detect_voice_activity(frames)
    assert flags.shape == (18,)
    assert flags.dtype == bool
    assert np.all(flags[:6])


def test_silence_after_hangover_is_unvoiced():
    frames = _frames()
    flags = detect_voice_activity(frames, hangover_frames=2)
    assert flags[6] and flags[7]  # hangover keeps two silent frames voiced
    assert not np.any(flags[8:])


def test_zero_hangover_ends_exactly_at_silence():
    frames = _frames()
    flags = detect_voice_activity(frames, hangover_frames=0)
    assert flags[5]
    assert not np.any(flags[6:])


def test_high_threshold_requires_full_scale_frames():
    frames = _frames()
    flags = detect_voice_activity(frames, threshold_db=-0.5, hangover_frames=0)
    expected = [0, 0, 1, 1] + [0] * 14  # only frames 2 and 3 are fully covered
    assert list(flags.astype(int)) == expected
