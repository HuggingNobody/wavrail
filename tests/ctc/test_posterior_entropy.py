"""validate_posteriors and frame_entropy contracts."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.ctc import frame_entropy, validate_posteriors
from wavrail.errors import DecodingError


def test_one_hot_rows_have_zero_entropy():
    p = np.eye(4, dtype=np.float64)
    entropy = frame_entropy(p)
    assert entropy.shape == (4,)
    assert entropy.dtype == np.float32
    np.testing.assert_allclose(entropy, 0.0, atol=1e-6)


def test_uniform_rows_have_max_entropy():
    p = np.full((3, 4), 0.25)
    np.testing.assert_allclose(frame_entropy(p), 1.0, atol=1e-6)


def test_mixed_row_sits_between_extremes():
    p = np.array([[0.7, 0.1, 0.1, 0.1]])
    value = float(frame_entropy(p)[0])
    assert 0.0 < value < 1.0


def test_single_column_is_zero_entropy():
    p = np.ones((5, 1))
    np.testing.assert_allclose(frame_entropy(p), 0.0, atol=1e-7)


def test_validation_rejects_bad_input():
    with pytest.raises(DecodingError):
        validate_posteriors(np.zeros(4))  # 1-D
    with pytest.raises(DecodingError):
        validate_posteriors(np.zeros((0, 3)))  # no frames
    with pytest.raises(DecodingError):
        validate_posteriors(np.array([[0.5, -0.5]]))  # negative
    with pytest.raises(DecodingError):
        validate_posteriors(np.array([[0.5, 0.25]]))  # rows must sum to 1
