"""Log-mel spectrogram extraction.

Combines :func:`wavrail.features.frames.frame_signal`,
:func:`wavrail.features.windows.hann_window`,
:func:`wavrail.features.preemphasis.pre_emphasize`, and
:func:`wavrail.features.mel.mel_filterbank` into a single
end-to-end helper. The output is a ``float32`` array of shape
``(num_frames, num_mels)``.
"""

from __future__ import annotations

import numpy as np

from wavrail.errors import FeatureError
from wavrail.features.frames import frame_signal
from wavrail.features.mel import mel_filterbank
from wavrail.features.preemphasis import pre_emphasize
from wavrail.features.spectrum import power_spectrum
from wavrail.features.windows import hann_window
from wavrail.types import AudioArray, FeatureArray, FrameConfig, MelConfig


def log_mel_spectrogram(
    samples: AudioArray,
    sample_rate: int,
    frame_config: FrameConfig,
    mel_config: MelConfig,
    preemphasis_coefficient: float = 0.97,
    log_floor: float | None = None,
) -> FeatureArray:
    """Compute log-mel features for the entire signal."""
    if mel_config.sample_rate != sample_rate:
        raise FeatureError("mel_config.sample_rate must match sample_rate")
    if mel_config.fft_size < frame_config.frame_length:
        raise FeatureError("mel_config.fft_size must be >= frame_config.frame_length")
    emphasised = pre_emphasize(samples, coefficient=preemphasis_coefficient)
    frames = frame_signal(emphasised, frame_config)
    window = hann_window(frame_config.frame_length, symmetric=False)
    spectrum = power_spectrum(frames, fft_size=mel_config.fft_size, window=window)
    filterbank = mel_filterbank(mel_config)
    mel_energies = spectrum @ filterbank.T
    floor = log_floor if log_floor is not None else mel_config.log_floor
    return np.log(np.maximum(mel_energies, floor)).astype(np.float32)


__all__ = ["log_mel_spectrogram"]
