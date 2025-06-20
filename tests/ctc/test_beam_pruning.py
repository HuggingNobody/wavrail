"""Threshold pruning relative to the running beam best."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.ctc import PrefixBeamSearch
from wavrail.errors import DecodingError
from wavrail.vocab import Vocab


def _vocab() -> Vocab:
    return Vocab.from_labels(["<blank>", "a", "b"])


def _random_log_posteriors(seed: int, frames: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probs = rng.random((frames, 3)) + 0.01
    probs /= probs.sum(axis=1, keepdims=True)
    return np.log(probs)


def _peaked(sequence: list[int]) -> np.ndarray:
    arr = np.full((len(sequence), 3), -10.0)
    for t, idx in enumerate(sequence):
        arr[t, idx] = 0.0
    return arr


def test_looser_threshold_never_worsens_best() -> None:
    for seed in range(8):
        lp = _random_log_posteriors(seed)
        tight = PrefixBeamSearch(vocab=_vocab(), beam=6, pruning_threshold=-1.0).decode(lp)
        loose = PrefixBeamSearch(vocab=_vocab(), beam=6, pruning_threshold=-30.0).decode(lp)
        assert loose.confidence >= tight.confidence - 1e-12


def test_rejects_positive_threshold() -> None:
    with pytest.raises(DecodingError, match="pruning_threshold"):
        PrefixBeamSearch(vocab=_vocab(), pruning_threshold=0.5)


def test_peaked_result_stable_across_thresholds() -> None:
    lp = _peaked([1, 0, 2, 0, 1])
    for threshold in (-0.5, -5.0, -20.0):
        decoder = PrefixBeamSearch(vocab=_vocab(), beam=6, pruning_threshold=threshold)
        assert decoder.decode(lp).tokens == (1, 2, 1)
