"""DCT-II matrix construction: orthogonality and known values."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.errors import FeatureError
from wavrail.features import dct2_matrix


def test_square_matrix_is_orthonormal():
    matrix = dct2_matrix(8, 8)
    assert matrix.shape == (8, 8)
    assert matrix.dtype == np.float64
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(8), atol=1e-12)


def test_dc_row_is_constant():
    matrix = dct2_matrix(4, 4)
    np.testing.assert_allclose(matrix[0], np.full(4, 0.5), atol=1e-12)


def test_rectangular_shape_truncates_rows():
    matrix = dct2_matrix(3, 40)
    assert matrix.shape == (3, 40)
    np.testing.assert_allclose(matrix, dct2_matrix(40, 40)[:3], atol=1e-12)


def test_rows_have_unit_norm():
    matrix = dct2_matrix(5, 17)
    norms = np.linalg.norm(matrix, axis=1)
    np.testing.assert_allclose(norms, np.ones(5), atol=1e-12)


def test_rejects_non_positive_dims():
    with pytest.raises(FeatureError):
        dct2_matrix(0, 4)
    with pytest.raises(FeatureError):
        dct2_matrix(4, 0)
