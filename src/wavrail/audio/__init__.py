"""Audio I/O and synthetic waveform generation.

This subpackage is the lowest layer of wavrail. It depends only on
NumPy and the standard library. Higher layers (``features``,
``vocab``, ``ctc``, ``streaming``) build on top of the primitives
exposed here.

Public entry points:

- :func:`read_wav` and :func:`write_wav` for PCM16 / PCM32 / Float32
  WAV files.
- :func:`inspect_wav` for header-only metadata.
- :func:`resample`, :func:`downmix_to_mono`, :func:`clipping_detected`,
  :func:`silence_ratio`.
- Synthetic generators: :func:`synthesize_sine`, :func:`synthesize_chirp`,
  :func:`synthesize_pulse`, :func:`synthesize_noise`,
  :func:`synthesize_formant_stack`.
"""

from wavrail.audio.downmix import downmix_to_mono
from wavrail.audio.info import inspect_wav
from wavrail.audio.quality import clipping_detected, silence_ratio
from wavrail.audio.resample import resample
from wavrail.audio.synthetic import (
    synthesize_chirp,
    synthesize_formant_stack,
    synthesize_noise,
    synthesize_pulse,
    synthesize_sine,
)
from wavrail.audio.wav import read_wav, write_wav

__all__ = [
    "clipping_detected",
    "downmix_to_mono",
    "inspect_wav",
    "read_wav",
    "resample",
    "silence_ratio",
    "synthesize_chirp",
    "synthesize_formant_stack",
    "synthesize_noise",
    "synthesize_pulse",
    "synthesize_sine",
    "write_wav",
]
