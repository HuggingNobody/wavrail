"""Spectral flux on stationary and changing spectra."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.errors import FeatureError
from wavrail.features import spectral_flux


def test_stationary_spectrum_has_zero_flux_after_onset():
    spectra = np.tile(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), (4, 1))
    flux = spectral_flux(spectra)
    assert flux.shape == (4,)
    assert flux.dtype == np.float32
    assert flux[0] == pytest.approx(6.0)  # first frame compares against silence
    assert np.allclose(flux[1:], 0.0)


def test_energy_jump_shows_up_in_flux():
    spectra = np.zeros((3, 4), dtype=np.float32)
    spectra[0, 0] = 1.0
    spectra[2, 0] = 5.0
    flux = spectral_flux(spectra)
    assert flux[0] == pytest.approx(1.0)
    assert flux[1] == pytest.approx(0.0)  # the drop is ignored by half-wave rectification
    assert flux[2] == pytest.approx(5.0)


def test_half_wave_false_uses_absolute_differences():
    spectra = np.array([[4.0], [1.0], [3.0]], dtype=np.float32)
    half = spectral_flux(spectra, half_wave=True)
    full = spectral_flux(spectra, half_wave=False)
    assert half[1] == pytest.approx(0.0)
    assert full[1] == pytest.approx(3.0)
    assert full[2] == pytest.approx(2.0)


def test_rejects_bad_input():
    with pytest.raises(FeatureError):
        spectral_flux(np.zeros(5, dtype=np.float32))
    negative = np.array([[-1.0, 0.0]], dtype=np.float32)
    with pytest.raises(FeatureError):
        spectral_flux(negative)
