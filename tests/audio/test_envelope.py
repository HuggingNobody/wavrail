"""Raised-cosine fade envelope behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.audio import apply_fade
from wavrail.errors import AudioError


def test_default_fades_are_identity():
    arr = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    assert np.array_equal(apply_fade(arr, 16000), arr)


def test_edges_reach_zero():
    ones = np.ones(160, dtype=np.float32)
    faded = apply_fade(ones, 16000, fade_in_s=0.001, fade_out_s=0.001)
    assert faded[0] == pytest.approx(0.0, abs=1e-6)
    assert faded[-1] == pytest.approx(0.0, abs=1e-6)
    assert faded[80] == pytest.approx(1.0)


def test_fade_in_envelope_is_monotone():
    ones = np.ones(64, dtype=np.float32)
    faded = apply_fade(ones, 16000, fade_in_s=0.001)  # 16-sample ramp
    ramp = faded[:16]
    assert np.all(np.diff(ramp) >= 0.0)
    assert ramp[-1] == pytest.approx(1.0)


def test_overlapping_fades_multiply_and_stay_bounded():
    ones = np.ones(32, dtype=np.float32)
    faded = apply_fade(ones, 16000, fade_in_s=0.0015, fade_out_s=0.0015)  # 24+24 > 32
    assert faded.shape == (32,)
    assert np.all(np.abs(faded) <= 1.0)
    assert faded[16] < 1.0  # both ramps are active mid-signal
    assert faded[0] == pytest.approx(0.0, abs=1e-6)
    assert faded[-1] == pytest.approx(0.0, abs=1e-6)


def test_multichannel_fades_along_last_axis():
    stereo = np.ones((2, 64), dtype=np.float32)
    faded = apply_fade(stereo, 16000, fade_in_s=0.0005)
    assert faded.shape == (2, 64)
    assert faded.dtype == np.float32
    assert np.allclose(faded[0], faded[1])


def test_validation_errors():
    ones = np.ones(16, dtype=np.float32)
    with pytest.raises(AudioError):
        apply_fade(ones, 16000, fade_in_s=-1.0)
    with pytest.raises(AudioError):
        apply_fade(ones, 16000, fade_in_s=0.01)  # 160 samples > 16
    with pytest.raises(AudioError):
        apply_fade(ones, 16000, fade_out_s=0.01)
    with pytest.raises(AudioError):
        apply_fade(np.zeros(0, dtype=np.float32), 16000, fade_in_s=0.001)
    with pytest.raises(AudioError):
        apply_fade(ones, 0)
