"""Frame energy and log frame energy on hand-computed values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wavrail.errors import FeatureError
from wavrail.features import frame_energy, log_frame_energy


def test_energy_matches_hand_computed():
    frames = np.array([[3.0, 4.0], [0.0, 0.5]], dtype=np.float32)
    energy = frame_energy(frames)
    assert energy.shape == (2,)
    assert energy.dtype == np.float32
    assert energy[0] == pytest.approx(25.0)
    assert energy[1] == pytest.approx(0.25)


def test_log_energy_floor_keeps_silence_finite():
    frames = np.zeros((2, 8), dtype=np.float32)
    logged = log_frame_energy(frames, floor=1e-10)
    assert np.all(np.isfinite(logged))
    assert logged[0] == pytest.approx(math.log(1e-10), rel=1e-5)


def test_log_energy_is_monotone_in_energy():
    frames = np.array([[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    logged = log_frame_energy(frames)
    assert np.all(np.diff(logged) > 0.0)


def test_rejects_bad_input():
    with pytest.raises(FeatureError):
        frame_energy(np.zeros(8, dtype=np.float32))
    with pytest.raises(FeatureError):
        log_frame_energy(np.zeros((1, 8), dtype=np.float32), floor=0.0)
