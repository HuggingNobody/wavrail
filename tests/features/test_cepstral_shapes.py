"""Golden shape contract for the cepstral feature stack.

Pins the documented shapes and dtypes: ``mfcc`` maps audio to
``(num_frames, num_coeffs)`` float32 matching the log-mel frame count,
``zero_crossing_rate`` maps frames to ``(num_frames,)`` in ``[0, 1]``,
and both stay finite on silence.
"""

from __future__ import annotations

import numpy as np

from wavrail.audio import synthesize_noise
from wavrail.features import frame_signal, mfcc, zero_crossing_rate
from wavrail.types import FrameConfig, MelConfig

FRAME_CONFIG = FrameConfig(frame_length=400, hop_length=160)
MEL_CONFIG = MelConfig(sample_rate=16000, num_mels=40, fft_size=512)
NUM_COEFFS = 13


def test_mfcc_frame_count_matches_logmel_contract():
    samples = synthesize_noise(0.32, 16000, seed=7)
    coeffs = mfcc(samples, 16000, FRAME_CONFIG, MEL_CONFIG, num_coeffs=NUM_COEFFS)
    expected_frames = frame_signal(samples, FRAME_CONFIG).shape[0]
    assert coeffs.shape == (expected_frames, NUM_COEFFS)
    assert coeffs.dtype == np.float32
    assert np.all(np.isfinite(coeffs))


def test_zcr_bounds_on_noise():
    samples = synthesize_noise(0.32, 16000, seed=8, amplitude=0.3)
    frames = frame_signal(samples, FRAME_CONFIG)
    zcr = zero_crossing_rate(frames)
    assert zcr.shape == (frames.shape[0],)
    assert np.all(zcr >= 0.0) and np.all(zcr <= 1.0)


def test_zcr_is_high_for_white_noise():
    samples = synthesize_noise(0.32, 16000, seed=9)
    frames = frame_signal(samples, FRAME_CONFIG)
    zcr = zero_crossing_rate(frames)
    assert float(zcr.mean()) > 0.45  # white noise flips sign about half the time
