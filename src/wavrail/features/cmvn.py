"""Cepstral mean and variance normalisation.

:func:`cmvn_normalize` normalises a feature matrix using precomputed
statistics. :class:`StreamingCmvn` accumulates running sums so that
the statistics can be estimated on the same stream that is being
decoded.
"""

from __future__ import annotations

import numpy as np

from wavrail.errors import FeatureError
from wavrail.types import FeatureArray


def cmvn_normalize(features: FeatureArray, mean: np.ndarray, std: np.ndarray) -> FeatureArray:
    """Apply ``(features - mean) / std`` per column."""
    if mean.ndim != 1 or std.ndim != 1:
        raise FeatureError("cmvn statistics must be 1-D")
    if mean.shape[0] != std.shape[0]:
        raise FeatureError("mean and std must have the same length")
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape[1] != mean.shape[0]:
            raise FeatureError("feature width does not match cmvn statistics length")
        return ((arr - mean) / np.maximum(std, 1e-10)).astype(np.float32)
    raise FeatureError("features must be a 2-D array")


class StreamingCmvn:
    """Online accumulator for mean and variance statistics.

    The accumulator consumes frames one batch at a time and
    exposes the current statistics via :meth:`mean` and
    :meth:`inv_std`. This is a streaming alternative to a full
    utterance pass; it does not discard old frames.
    """

    def __init__(self, num_features: int) -> None:
        if num_features <= 0:
            raise FeatureError("num_features must be positive")
        self._num_features = num_features
        self._count = 0
        self._sum = np.zeros(num_features, dtype=np.float64)
        self._sumsq = np.zeros(num_features, dtype=np.float64)

    def update(self, frames: FeatureArray) -> None:
        arr = np.asarray(frames, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self._num_features:
            raise FeatureError("frames must be (batch, num_features)")
        self._count += arr.shape[0]
        self._sum += arr.sum(axis=0)
        self._sumsq += (arr * arr).sum(axis=0)

    @property
    def count(self) -> int:
        return self._count

    def mean(self) -> np.ndarray:
        if self._count == 0:
            return np.zeros(self._num_features, dtype=np.float32)
        return (self._sum / self._count).astype(np.float32)

    def inv_std(self) -> np.ndarray:
        if self._count == 0:
            return np.ones(self._num_features, dtype=np.float32)
        variance = self._sumsq / self._count - (self._sum / self._count) ** 2
        return (1.0 / np.sqrt(np.maximum(variance, 1e-10))).astype(np.float32)


__all__ = ["StreamingCmvn", "cmvn_normalize"]
