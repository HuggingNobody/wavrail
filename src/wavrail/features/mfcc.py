"""Mel-frequency cepstral coefficients (MFCC).

MFCCs compress a log-mel spectrogram into a small set of cepstral
coefficients per frame: the log-mel rows are projected onto the first
``num_coeffs`` DCT-II basis rows and then optionally liftered. Because
the input is already log-scaled, coefficient 0 tracks overall log
energy while the remaining coefficients capture the spectral envelope
shape.

The pipeline reuses :func:`wavrail.features.log_mel_spectrogram` for
framing, windowing, pre-emphasis, and the mel filterbank, so every
tolerance and contract documented there applies unchanged.
"""

from __future__ import annotations

from wavrail.features.dct import apply_dct
from wavrail.features.lifter import apply_lifter
from wavrail.features.logmel import log_mel_spectrogram
from wavrail.types import AudioArray, FeatureArray, FrameConfig, MelConfig


def mfcc_from_logmel(
    logmel: FeatureArray,
    num_coeffs: int = 13,
    lifter: int = 22,
) -> FeatureArray:
    """Return MFCCs for a precomputed log-mel spectrogram.

    ``logmel`` has shape ``(num_frames, num_mels)``; the result has
    shape ``(num_frames, num_coeffs)`` as ``float32``. ``lifter = 0``
    skips liftering.
    """
    coeffs = apply_dct(logmel, num_coeffs)
    return apply_lifter(coeffs, lifter)


def mfcc(
    samples: AudioArray,
    sample_rate: int,
    frame_config: FrameConfig,
    mel_config: MelConfig,
    num_coeffs: int = 13,
    lifter: int = 22,
    preemphasis_coefficient: float = 0.97,
    log_floor: float | None = None,
) -> FeatureArray:
    """Compute MFCCs directly from audio samples.

    Equivalent to ``mfcc_from_logmel(log_mel_spectrogram(...))`` with
    the same arguments; kept as a separate entry point so callers do
    not have to thread the intermediate spectrogram through.
    """
    logmel = log_mel_spectrogram(
        samples,
        sample_rate,
        frame_config,
        mel_config,
        preemphasis_coefficient=preemphasis_coefficient,
        log_floor=log_floor,
    )
    return mfcc_from_logmel(logmel, num_coeffs=num_coeffs, lifter=lifter)


__all__ = ["mfcc", "mfcc_from_logmel"]
