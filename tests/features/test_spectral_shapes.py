"""Golden shape contract for the spectral descriptor stack.

Every descriptor maps a ``(num_frames, num_bins)`` power spectrum to
a ``(num_frames,)`` float32 vector of finite values, including on
all-silence input where the documented zero convention applies.
"""

from __future__ import annotations

import numpy as np

from wavrail.features import (
    spectral_bandwidth,
    spectral_centroid,
    spectral_flatness,
    spectral_flux,
    spectral_rolloff,
)

NUM_FRAMES = 7
NUM_BINS = 33  # n_fft = 64


def _descriptors(spectra: np.ndarray) -> list[np.ndarray]:
    return [
        spectral_centroid(spectra, 16000),
        spectral_bandwidth(spectra, 16000),
        spectral_rolloff(spectra, 16000),
        spectral_flatness(spectra, 16000),
        spectral_flux(spectra),
    ]


def test_descriptor_shapes_and_dtypes():
    rng = np.random.default_rng(11)
    spectra = rng.random((NUM_FRAMES, NUM_BINS)).astype(np.float32)
    for values in _descriptors(spectra):
        assert values.shape == (NUM_FRAMES,)
        assert values.dtype == np.float32
        assert np.all(np.isfinite(values))


def test_silence_stack_is_all_zero():
    spectra = np.zeros((NUM_FRAMES, NUM_BINS), dtype=np.float32)
    for values in _descriptors(spectra):
        assert np.all(values == 0.0)


def test_bounds_hold_on_random_spectra():
    rng = np.random.default_rng(12)
    spectra = rng.random((NUM_FRAMES, NUM_BINS)).astype(np.float32)
    centroid = spectral_centroid(spectra, 16000)
    rolloff = spectral_rolloff(spectra, 16000, percent=0.99)
    flatness = spectral_flatness(spectra, 16000)
    assert np.all(centroid >= 0.0) and np.all(centroid <= 8000.0)
    assert np.all(rolloff >= 0.0) and np.all(rolloff <= 8000.0)
    assert np.all(flatness >= 0.0) and np.all(flatness <= 1.0)
