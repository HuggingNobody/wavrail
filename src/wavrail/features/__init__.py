"""Signal processing features for speech recognition.

This subpackage builds on the audio primitives exposed by
:mod:`wavrail.audio`. It provides frame blocking, window functions,
pre-emphasis, the power spectrum, mel filterbank construction,
log-mel features, streaming cepstral mean and variance
normalisation (CMVN), delta derivatives, spectral descriptors,
frame energy, autocorrelation pitch estimates, an energy-based
voice-activity detector, MFCCs with cepstral liftering, and the
zero-crossing rate.

Public entry points are added incrementally as the corresponding
modules land; see ``__all__`` for the current surface.
"""

from wavrail.features.cmvn import StreamingCmvn, cmvn_normalize
from wavrail.features.dct import apply_dct, dct2_matrix
from wavrail.features.delta import compute_deltas
from wavrail.features.energy import (
    frame_energy,
    log_frame_energy,
    zero_crossing_rate,
)
from wavrail.features.frames import frame_signal, num_frames
from wavrail.features.lifter import apply_lifter, cepstral_lifter
from wavrail.features.logmel import log_mel_spectrogram
from wavrail.features.mel import hz_to_mel, mel_filterbank, mel_to_hz
from wavrail.features.mfcc import mfcc, mfcc_from_logmel
from wavrail.features.pitch import PitchEstimate, estimate_pitch_acf
from wavrail.features.preemphasis import pre_emphasize
from wavrail.features.spectral import (
    spectral_bandwidth,
    spectral_centroid,
    spectral_flatness,
    spectral_flux,
    spectral_rolloff,
)
from wavrail.features.spectrum import power_spectrum
from wavrail.features.vad import EnergyVad, detect_voice_activity
from wavrail.features.windows import (
    blackman_window,
    hamming_window,
    hann_window,
    rectangular_window,
)

__all__ = [
    "EnergyVad",
    "PitchEstimate",
    "StreamingCmvn",
    "apply_dct",
    "apply_lifter",
    "blackman_window",
    "cepstral_lifter",
    "cmvn_normalize",
    "compute_deltas",
    "dct2_matrix",
    "detect_voice_activity",
    "estimate_pitch_acf",
    "frame_energy",
    "frame_signal",
    "hamming_window",
    "hann_window",
    "hz_to_mel",
    "log_frame_energy",
    "log_mel_spectrogram",
    "mel_filterbank",
    "mel_to_hz",
    "mfcc",
    "mfcc_from_logmel",
    "num_frames",
    "power_spectrum",
    "pre_emphasize",
    "rectangular_window",
    "spectral_bandwidth",
    "spectral_centroid",
    "spectral_flatness",
    "spectral_flux",
    "spectral_rolloff",
    "zero_crossing_rate",
]
