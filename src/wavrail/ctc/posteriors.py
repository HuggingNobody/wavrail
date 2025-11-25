"""Posterior inspection and transformation utilities.

Helpers for the ``(num_frames, vocab_size)`` posterior matrices that
the decoders consume:

- :func:`validate_posteriors` enforces the probability contract
  (finite, non-negative, rows sum to 1 within ``atol``).
- :func:`frame_entropy` reports the Shannon entropy of each frame
  normalised to ``[0, 1]``: 0 for one-hot rows, 1 for uniform rows.

Entropy and the confidence helpers built on it describe decoder
*uncertainty*, not word accuracy; a confident frame can still be
wrong. The normalisation divides by ``log(vocab_size)`` so values are
comparable across vocabularies.
"""

from __future__ import annotations

import numpy as np

from wavrail.errors import DecodingError


def _as_2d(posteriors: np.ndarray, name: str = "posteriors") -> np.ndarray:
    arr = np.asarray(posteriors, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise DecodingError(f"{name} must be a non-empty 2-D (num_frames, vocab_size) array")
    return arr


def validate_posteriors(posteriors: np.ndarray, atol: float = 1e-4) -> np.ndarray:
    """Check the probability contract and return the ``float64`` view."""
    arr = _as_2d(posteriors)
    if not np.all(np.isfinite(arr)):
        raise DecodingError("posteriors must be finite")
    if np.any(arr < 0.0):
        raise DecodingError("posteriors must be non-negative")
    if not np.allclose(arr.sum(axis=1), 1.0, atol=atol):
        raise DecodingError("posterior rows must sum to 1 within tolerance")
    return arr


def frame_entropy(posteriors: np.ndarray) -> np.ndarray:
    """Return the normalised Shannon entropy of each frame in ``[0, 1]``."""
    arr = validate_posteriors(posteriors)
    num_classes = arr.shape[1]
    if num_classes == 1:
        return np.zeros(arr.shape[0], dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(arr > 0.0, arr * np.log(arr), 0.0)
    entropy = -terms.sum(axis=1) / np.log(num_classes)
    return np.clip(entropy, 0.0, 1.0).astype(np.float32)


def frame_confidence(posteriors: np.ndarray) -> np.ndarray:
    """Return the max probability of each frame as ``float32``.

    This is the confidence a greedy decoding step would carry at that
    frame: the posterior of the argmax label. Like entropy, it scores
    decoder certainty, not correctness.
    """
    arr = validate_posteriors(posteriors)
    return arr.max(axis=1).astype(np.float32)


__all__ = ["frame_confidence", "frame_entropy", "validate_posteriors"]
