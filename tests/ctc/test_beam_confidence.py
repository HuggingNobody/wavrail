"""Confidence semantics of the prefix beam decoder.

Confidence is ``exp(score / T)`` for the prefix log-probability;
it is a relative ordering signal, not a calibrated probability.
"""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.ctc import PrefixBeamSearch
from wavrail.vocab import Vocab


def _vocab() -> Vocab:
    return Vocab.from_labels(["<blank>", "a", "b"])


def _peaked(sequence: list[int]) -> np.ndarray:
    arr = np.full((len(sequence), 3), -10.0)
    for t, idx in enumerate(sequence):
        arr[t, idx] = 0.0
    return arr


def test_confidence_within_unit_interval() -> None:
    for seed in range(6):
        rng = np.random.default_rng(seed)
        probs = rng.random((5, 3)) + 0.01
        probs /= probs.sum(axis=1, keepdims=True)
        hyp = PrefixBeamSearch(vocab=_vocab(), beam=8).decode(np.log(probs))
        assert 0.0 <= hyp.confidence <= 1.0


def test_confidence_near_one_for_peaked_input() -> None:
    hyp = PrefixBeamSearch(vocab=_vocab(), beam=4).decode(_peaked([1, 0, 2]))
    assert hyp.confidence == pytest.approx(1.0)


def test_confidence_clamped_for_unnormalized_input() -> None:
    # rows whose exp-mass exceeds one are invalid posteriors; the
    # reported confidence must still respect the [0, 1] bound
    lp = np.array([[0.0, -5.0, -5.0], [0.0, -5.0, -5.0]])
    hyp = PrefixBeamSearch(vocab=_vocab(), beam=4).decode(lp)
    assert hyp.confidence <= 1.0


def test_confidence_zero_for_empty_input() -> None:
    hyp = PrefixBeamSearch(vocab=_vocab(), beam=4).decode(np.zeros((0, 3)))
    assert hyp.confidence == 0.0
