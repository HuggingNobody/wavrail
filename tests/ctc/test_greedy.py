"""Tests for :func:`wavrail.ctc.greedy_decode`."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.ctc import greedy_decode
from wavrail.errors import DecodingError
from wavrail.vocab import Vocab


def _vocab() -> Vocab:
    return Vocab.from_labels(["<blank>", "a", "b", "c"])


def _posterior(sequence: list[int], num_classes: int = 4) -> np.ndarray:
    arr = np.full((len(sequence), num_classes), 1e-3, dtype=np.float32)
    for t, idx in enumerate(sequence):
        arr[t, idx] = 1.0
    return arr / arr.sum(axis=1, keepdims=True)


def test_greedy_decode_basic_sequence() -> None:
    posteriors = _posterior([0, 1, 1, 0, 2, 3, 0])
    hyp = greedy_decode(posteriors, _vocab())
    assert hyp.tokens == (1, 2, 3)
    assert hyp.text == "abc"
    assert 0.0 <= hyp.confidence <= 1.0


def test_greedy_decode_all_blanks() -> None:
    posteriors = _posterior([0, 0, 0])
    hyp = greedy_decode(posteriors, _vocab())
    assert hyp.tokens == ()
    assert hyp.text == ""


def test_greedy_decode_confidence_high_for_perfect() -> None:
    posteriors = _posterior([1, 1, 1])
    hyp = greedy_decode(posteriors, _vocab())
    assert hyp.confidence > 0.99
    assert hyp.confidence <= 1.0


def test_greedy_decode_rejects_shape_mismatch() -> None:
    posteriors = np.zeros((3, 5), dtype=np.float32)
    with pytest.raises(DecodingError):
        greedy_decode(posteriors, _vocab())


def test_greedy_decode_rejects_non_2d() -> None:
    posteriors = np.zeros((3, 4, 5), dtype=np.float32)
    with pytest.raises(DecodingError):
        greedy_decode(posteriors, _vocab())
