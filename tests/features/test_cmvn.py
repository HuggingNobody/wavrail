"""Tests for the CMVN module."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.errors import FeatureError
from wavrail.features.cmvn import StreamingCmvn, cmvn_normalize


def test_cmvn_normalize_zero_mean_unit_var() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(2.0, 0.5, size=(100, 8)).astype(np.float32)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    out = cmvn_normalize(features, mean=mean, std=std)
    assert out.shape == features.shape
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-4)


def test_cmvn_normalize_rejects_shape_mismatch() -> None:
    features = np.zeros((4, 6), dtype=np.float32)
    with pytest.raises(FeatureError):
        cmvn_normalize(features, mean=np.zeros(5), std=np.ones(5))


def test_streaming_cmvn_matches_offline() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(0.0, 1.0, size=(50, 4)).astype(np.float32)
    expected_mean = features.mean(axis=0)
    expected_std = features.std(axis=0)
    accumulator = StreamingCmvn(num_features=4)
    batch_size = 10
    for start in range(0, features.shape[0], batch_size):
        accumulator.update(features[start : start + batch_size])
    np.testing.assert_allclose(accumulator.mean(), expected_mean, atol=1e-4)
    np.testing.assert_allclose(
        accumulator.inv_std(), 1.0 / np.maximum(expected_std, 1e-10), atol=1e-3
    )


def test_streaming_cmvn_handles_empty_state() -> None:
    accumulator = StreamingCmvn(num_features=3)
    assert accumulator.count == 0
    np.testing.assert_array_equal(accumulator.mean(), np.zeros(3))
    np.testing.assert_array_equal(accumulator.inv_std(), np.ones(3))


def test_streaming_cmvn_rejects_wrong_width() -> None:
    accumulator = StreamingCmvn(num_features=3)
    with pytest.raises(FeatureError):
        accumulator.update(np.zeros((1, 4), dtype=np.float32))
