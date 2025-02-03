"""Audio I/O and synthetic waveform generation.

This subpackage is the lowest layer of wavrail. It depends only on
NumPy and the standard library. Higher layers (``features``,
``vocab``, ``ctc``, ``streaming``) build on top of the primitives
exposed here.

Public entry points are added incrementally as the corresponding
codec, transform, and generator modules land. The package docstring
documents the planned surface; check ``__all__`` for what is actually
exported by a given release.
"""

from wavrail.audio.info import inspect_wav
from wavrail.audio.synthetic import synthesize_chirp, synthesize_pulse, synthesize_sine
from wavrail.audio.wav import read_wav, write_wav

__all__ = [
    "inspect_wav",
    "read_wav",
    "synthesize_chirp",
    "synthesize_pulse",
    "synthesize_sine",
    "write_wav",
]
