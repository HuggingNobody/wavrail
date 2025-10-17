"""mfcc_from_logmel on controlled log-mel inputs."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.errors import FeatureError
from wavrail.features import apply_dct, mfcc_from_logmel


def test_shape_and_dtype():
    logmel = np.random.default_rng(2).standard_normal((9, 40)).astype(np.float32)
    coeffs = mfcc_from_logmel(logmel, num_coeffs=13)
    assert coeffs.shape == (9, 13)
    assert coeffs.dtype == np.float32


def test_matches_dct_then_lifter():
    logmel = np.random.default_rng(3).standard_normal((5, 23)).astype(np.float32)
    coeffs = mfcc_from_logmel(logmel, num_coeffs=7, lifter=22)
    manual = apply_dct(logmel, 7)
    weights = 1.0 + 11.0 * np.sin(np.pi * np.arange(7) / 22.0)
    weights[0] = 1.0
    np.testing.assert_allclose(coeffs, manual * weights, atol=1e-5)


def test_zero_lifter_skips_reweighting():
    logmel = np.random.default_rng(4).standard_normal((4, 16)).astype(np.float32)
    plain = mfcc_from_logmel(logmel, num_coeffs=8, lifter=0)
    np.testing.assert_allclose(plain, apply_dct(logmel, 8), atol=1e-6)


def test_rejects_too_many_coeffs():
    with pytest.raises(FeatureError):
        mfcc_from_logmel(np.zeros((2, 8), dtype=np.float32), num_coeffs=9)
