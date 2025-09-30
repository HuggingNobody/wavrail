"""apply_dct: shapes, DC semantics, and energy preservation."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.errors import FeatureError
from wavrail.features import apply_dct


def test_dc_coefficient_is_scaled_mean():
    features = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    coeffs = apply_dct(features, 1)
    assert coeffs.shape == (1, 1)
    assert coeffs.dtype == np.float32
    # row 0 has scale sqrt(1/4) = 0.5, so DC = 0.5 * sum = mean * sqrt(N)
    assert coeffs[0, 0] == pytest.approx(2.0, rel=1e-6)


def test_full_transform_preserves_energy():
    rng = np.random.default_rng(5)
    features = rng.standard_normal((6, 16))
    coeffs = apply_dct(features, 16)
    input_energy = np.sum(features * features)
    output_energy = float(np.sum(coeffs.astype(np.float64) ** 2))
    assert output_energy == pytest.approx(input_energy, rel=1e-5)


def test_operates_row_by_row():
    features = np.tile(np.array([[2.0, 0.0, 0.0, 0.0]], dtype=np.float32), (3, 1))
    coeffs = apply_dct(features, 4)
    assert coeffs.shape == (3, 4)
    np.testing.assert_allclose(coeffs[0], coeffs[1], atol=1e-6)
    np.testing.assert_allclose(coeffs[1], coeffs[2], atol=1e-6)


def test_rejects_bad_input():
    with pytest.raises(FeatureError):
        apply_dct(np.zeros(8, dtype=np.float32), 4)
    with pytest.raises(FeatureError):
        apply_dct(np.zeros((2, 4), dtype=np.float32), 5)  # more coeffs than inputs
