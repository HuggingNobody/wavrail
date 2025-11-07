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

from wavrail.audio.concat import concatenate
from wavrail.audio.downmix import downmix_to_mono
from wavrail.audio.energy import (
    crest_factor,
    from_dbfs,
    peak_amplitude,
    rms_energy,
    to_dbfs,
)
from wavrail.audio.envelope import apply_fade
from wavrail.audio.info import inspect_wav
from wavrail.audio.normalize import (
    apply_gain_db,
    dc_offset,
    normalize_peak,
    remove_dc_offset,
)
from wavrail.audio.quality import clipping_detected, silence_ratio
from wavrail.audio.resample import resample
from wavrail.audio.synthetic import (
    synthesize_chirp,
    synthesize_formant_stack,
    synthesize_multitone,
    synthesize_noise,
    synthesize_pulse,
    synthesize_silence,
    synthesize_sine,
)
from wavrail.audio.trim import trim_silence
from wavrail.audio.wav import read_wav, write_wav

__all__ = [
    "apply_fade",
    "apply_gain_db",
    "clipping_detected",
    "concatenate",
    "crest_factor",
    "dc_offset",
    "downmix_to_mono",
    "from_dbfs",
    "inspect_wav",
    "normalize_peak",
    "peak_amplitude",
    "read_wav",
    "remove_dc_offset",
    "resample",
    "rms_energy",
    "silence_ratio",
    "synthesize_chirp",
    "synthesize_formant_stack",
    "synthesize_multitone",
    "synthesize_noise",
    "synthesize_pulse",
    "synthesize_silence",
    "synthesize_sine",
    "to_dbfs",
    "trim_silence",
    "write_wav",
]
