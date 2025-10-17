"""End-to-end mfcc on synthetic fixtures."""

from __future__ import annotations

import numpy as np

from wavrail.audio import synthesize_silence, synthesize_sine
from wavrail.features import log_mel_spectrogram, mfcc, mfcc_from_logmel
from wavrail.types import FrameConfig, MelConfig

FRAME_CONFIG = FrameConfig(frame_length=400, hop_length=160)
MEL_CONFIG = MelConfig(sample_rate=16000, num_mels=40, fft_size=512)


def test_sine_mfcc_shapes_and_finiteness():
    samples = synthesize_sine(440.0, 0.25, 16000)
    coeffs = mfcc(samples, 16000, FRAME_CONFIG, MEL_CONFIG, num_coeffs=13)
    assert coeffs.ndim == 2
    assert coeffs.shape[1] == 13
    assert coeffs.dtype == np.float32
    assert np.all(np.isfinite(coeffs))


def test_equivalent_to_logmel_then_dct():
    samples = synthesize_sine(300.0, 0.2, 16000, amplitude=0.5)
    direct = mfcc(samples, 16000, FRAME_CONFIG, MEL_CONFIG, num_coeffs=10, lifter=0)
    logmel = log_mel_spectrogram(samples, 16000, FRAME_CONFIG, MEL_CONFIG)
    staged = mfcc_from_logmel(logmel, num_coeffs=10, lifter=0)
    np.testing.assert_allclose(direct, staged, atol=1e-5)


def test_silence_stays_finite():
    samples = synthesize_silence(0.2, 16000)
    coeffs = mfcc(samples, 16000, FRAME_CONFIG, MEL_CONFIG, num_coeffs=13)
    assert np.all(np.isfinite(coeffs))
    # constant log-mel rows -> only the DC coefficient is non-zero
    np.testing.assert_allclose(coeffs[:, 1:], np.zeros_like(coeffs[:, 1:]), atol=1e-3)
