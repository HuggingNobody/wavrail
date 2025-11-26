"""rescale_temperature: identity, sharpening, flattening, validation."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.ctc import rescale_temperature
from wavrail.errors import DecodingError


def _logp(seed: int = 0, frames: int = 5, classes: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = rng.random((frames, classes)) + 0.05
    p = rows / rows.sum(axis=1, keepdims=True)
    return np.log(p)


def test_identity_temperature_preserves_probabilities():
    logp = _logp()
    out = rescale_temperature(logp, 1.0)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.exp(out), np.exp(logp), atol=1e-5)


def test_low_temperature_sharpens():
    logp = _logp()
    out = rescale_temperature(logp, 0.5)
    before = float(np.exp(logp).max(axis=1).mean())
    after = float(np.exp(out).max(axis=1).mean())
    assert after > before


def test_high_temperature_flattens():
    out = rescale_temperature(_logp(), 100.0)
    np.testing.assert_allclose(np.exp(out), 0.25, atol=1e-2)


def test_output_rows_are_normalized():
    out = rescale_temperature(_logp(seed=3), 2.0)
    np.testing.assert_allclose(np.exp(out.astype(np.float64)).sum(axis=1), 1.0, atol=1e-5)


def test_validation():
    with pytest.raises(DecodingError):
        rescale_temperature(_logp(), 0.0)
    with pytest.raises(DecodingError):
        rescale_temperature(_logp(), -1.0)
    with pytest.raises(DecodingError):
        rescale_temperature(np.zeros(5), 1.0)  # 1-D
