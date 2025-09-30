"""Orthonormal DCT-II construction for cepstral feature transforms.

The discrete cosine transform of type II projects a real vector onto a
basis of shifted cosines. :func:`dct2_matrix` builds the matrix in the
orthonormal normalisation: row 0 (the DC coefficient) is scaled by
``sqrt(1 / N)`` and every other row by ``sqrt(2 / N)``, so a square
matrix satisfies ``M @ M.T == I`` and the transform preserves energy
(Parseval). MFCCs use the first ``num_coeffs`` rows of a square-ish
matrix to compress log-mel filterbank outputs.
"""

from __future__ import annotations

import numpy as np

from wavrail.errors import FeatureError
from wavrail.types import FeatureArray


def dct2_matrix(num_coeffs: int, num_inputs: int) -> np.ndarray:
    """Return the ``(num_coeffs, num_inputs)`` orthonormal DCT-II matrix.

    Row ``k`` evaluates ``cos(pi * (n + 0.5) * k / num_inputs)`` over
    ``n = 0 .. num_inputs - 1`` with the orthonormal scale described in
    the module docstring. The result is ``float64`` so downstream
    precision decisions stay with the caller.
    """
    if num_coeffs < 1 or num_inputs < 1:
        raise FeatureError("num_coeffs and num_inputs must be positive")
    k = np.arange(num_coeffs, dtype=np.float64).reshape(-1, 1)
    n = np.arange(num_inputs, dtype=np.float64).reshape(1, -1)
    matrix = np.cos(np.pi * (n + 0.5) * k / num_inputs)
    scale = np.full((num_coeffs, 1), np.sqrt(2.0 / num_inputs))
    scale[0] = np.sqrt(1.0 / num_inputs)
    return matrix * scale


def apply_dct(features: FeatureArray, num_coeffs: int) -> FeatureArray:
    """Project every row of ``features`` onto the first DCT-II basis rows.

    Consumes a 2-D array of shape ``(num_frames, num_inputs)`` and
    returns ``(num_frames, num_coeffs)`` as ``float32``. Requesting
    more coefficients than inputs is rejected: the DCT-II basis only
    spans ``num_inputs`` dimensions, so extra rows would be redundant
    or misleading.
    """
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim != 2:
        raise FeatureError("features must be a 2-D array of shape (num_frames, num_inputs)")
    if num_coeffs > arr.shape[1]:
        raise FeatureError(
            f"num_coeffs ({num_coeffs}) exceeds the number of inputs ({arr.shape[1]})"
        )
    matrix = dct2_matrix(num_coeffs, arr.shape[1])
    return (arr @ matrix.T).astype(np.float32)


__all__ = ["apply_dct", "dct2_matrix"]
