"""Signal processing features for speech recognition.

This subpackage builds on the audio primitives exposed by
:mod:`wavrail.audio`. It provides frame blocking, window functions,
pre-emphasis, the power spectrum, mel filterbank construction,
log-mel features, streaming cepstral mean and variance
normalisation (CMVN), and delta derivatives.

Public entry points are added incrementally as the corresponding
modules land; see ``__all__`` for the current surface.
"""

from wavrail.features.cmvn import StreamingCmvn, cmvn_normalize
from wavrail.features.delta import compute_deltas
from wavrail.features.frames import frame_signal, num_frames
from wavrail.features.logmel import log_mel_spectrogram
from wavrail.features.mel import hz_to_mel, mel_filterbank, mel_to_hz
from wavrail.features.preemphasis import pre_emphasize
from wavrail.features.spectral import (
    spectral_bandwidth,
    spectral_centroid,
    spectral_flatness,
    spectral_flux,
    spectral_rolloff,
)
from wavrail.features.spectrum import power_spectrum
from wavrail.features.windows import (
    blackman_window,
    hamming_window,
    hann_window,
    rectangular_window,
)

__all__ = [
    "StreamingCmvn",
    "blackman_window",
    "cmvn_normalize",
    "compute_deltas",
    "frame_signal",
    "hamming_window",
    "hann_window",
    "hz_to_mel",
    "log_mel_spectrogram",
    "mel_filterbank",
    "mel_to_hz",
    "num_frames",
    "power_spectrum",
    "pre_emphasize",
    "rectangular_window",
    "spectral_bandwidth",
    "spectral_centroid",
    "spectral_flatness",
    "spectral_flux",
    "spectral_rolloff",
]
