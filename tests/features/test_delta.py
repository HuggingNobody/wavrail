"""Tests for :func:`wavrail.features.compute_deltas`."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.errors import FeatureError
from wavrail.features.delta import compute_deltas


def test_deltas_of_constant_signal_are_zero() -> None:
    features = np.ones((32, 4), dtype=np.float32)
    deltas = compute_deltas(features, window=2)
    np.testing.assert_array_equal(deltas, np.zeros_like(features))


def test_deltas_of_linear_signal_are_constant() -> None:
    t = np.arange(16, dtype=np.float32)
    features = np.tile(t[:, None], (1, 3))
    deltas = compute_deltas(features, window=2)
    inner = deltas[4:-4]
    np.testing.assert_allclose(inner, np.ones_like(inner), atol=1e-5)


def test_deltas_preserve_shape_and_dtype() -> None:
    features = np.random.default_rng(0).normal(size=(20, 5)).astype(np.float32)
    deltas = compute_deltas(features, window=2)
    assert deltas.shape == features.shape
    assert deltas.dtype == np.float32


def test_deltas_rejects_invalid_inputs() -> None:
    with pytest.raises(FeatureError):
        compute_deltas(np.zeros((1, 4), dtype=np.float32))
    with pytest.raises(FeatureError):
        compute_deltas(np.zeros((4, 3), dtype=np.float32), window=0)
