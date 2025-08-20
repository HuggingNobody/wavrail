"""Crest factor and dBFS conversion behaviour."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wavrail.audio import crest_factor, from_dbfs, to_dbfs
from wavrail.errors import AudioError


def test_sine_crest_factor_is_sqrt_two():
    t = np.arange(1600, dtype=np.float64) / 16000.0
    tone = (0.7 * np.sin(2.0 * math.pi * 440.0 * t)).astype(np.float32)
    assert crest_factor(tone) == pytest.approx(math.sqrt(2.0), rel=1e-2)


def test_impulse_crest_factor_is_large():
    impulse = np.zeros(128, dtype=np.float32)
    impulse[0] = 1.0
    assert crest_factor(impulse) == pytest.approx(math.sqrt(128.0), rel=1e-3)


def test_silence_crest_factor_is_zero():
    assert crest_factor(np.zeros(8, dtype=np.float32)) == 0.0


def test_dbfs_endpoints():
    assert to_dbfs(1.0) == pytest.approx(0.0)
    assert to_dbfs(0.5) == pytest.approx(-6.0206, abs=1e-3)
    assert from_dbfs(0.0) == pytest.approx(1.0)


def test_dbfs_roundtrip():
    for value in (1.0, 0.5, 0.25, 0.01):
        assert from_dbfs(to_dbfs(value)) == pytest.approx(value, rel=1e-6)


def test_dbfs_rejects_out_of_range():
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(AudioError):
            to_dbfs(bad)
    for bad in (0.1, 6.0, float("nan")):
        with pytest.raises(AudioError):
            from_dbfs(bad)
    assert from_dbfs(float("-inf")) == 0.0
