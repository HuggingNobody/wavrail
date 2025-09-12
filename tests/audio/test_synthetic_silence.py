"""synthesize_silence produces exact zeros with the requested geometry."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.audio import synthesize_silence
from wavrail.errors import AudioError


def test_all_zero_output():
    out = synthesize_silence(0.01, 16000)
    assert out.shape == (160,)
    assert out.dtype == np.float32
    assert np.all(out == 0.0)


def test_length_follows_sample_rate():
    assert synthesize_silence(0.5, 8000).shape == (4000,)
    assert synthesize_silence(0.5, 22050).shape == (11025,)


def test_validation():
    with pytest.raises(AudioError):
        synthesize_silence(0.0, 16000)
    with pytest.raises(AudioError):
        synthesize_silence(-0.1, 16000)
    with pytest.raises(AudioError):
        synthesize_silence(0.1, 0)
