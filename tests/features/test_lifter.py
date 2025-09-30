"""Cepstral lifter weights and application."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.errors import FeatureError
from wavrail.features import apply_lifter, cepstral_lifter


def test_zero_lifter_is_identity():
    weights = cepstral_lifter(5, lifter=0)
    np.testing.assert_allclose(weights, np.ones(5))
    coeffs = np.arange(10, dtype=np.float32).reshape(2, 5)
    out = apply_lifter(coeffs, lifter=0)
    np.testing.assert_allclose(out, coeffs, atol=1e-6)


def test_htk_formula_midpoint():
    weights = cepstral_lifter(12, lifter=22)
    assert weights[0] == pytest.approx(1.0)
    # k = 11 -> 1 + 11 * sin(pi / 2) = 12
    assert weights[11] == pytest.approx(12.0, rel=1e-9)


def test_weights_are_float64_and_shaped():
    weights = cepstral_lifter(13, lifter=22)
    assert weights.shape == (13,)
    assert weights.dtype == np.float64


def test_apply_lifter_scales_rows():
    coeffs = np.ones((2, 4), dtype=np.float32)
    out = apply_lifter(coeffs, lifter=0)
    assert out.dtype == np.float32
    out22 = apply_lifter(coeffs, lifter=22)
    expected = cepstral_lifter(4, lifter=22)
    np.testing.assert_allclose(out22[0], expected, atol=1e-6)


def test_validation():
    with pytest.raises(FeatureError):
        cepstral_lifter(0)
    with pytest.raises(FeatureError):
        cepstral_lifter(4, lifter=-1)
    with pytest.raises(FeatureError):
        apply_lifter(np.zeros(4, dtype=np.float32))
