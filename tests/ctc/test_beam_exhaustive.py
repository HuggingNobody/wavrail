"""Exhaustive small-case checks against brute-force path enumeration.

A wide beam without threshold pruning computes exact CTC prefix
probabilities, so decoder scores must equal an explicit sum over
every alignment path that collapses to each label sequence.
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest

from wavrail.ctc import PrefixBeamSearch
from wavrail.vocab import Vocab


def brute_force_label_scores(
    log_posteriors: np.ndarray, blank: int = 0
) -> dict[tuple[int, ...], float]:
    """Sum the probability mass of every alignment path per label."""
    frames, vocab_size = log_posteriors.shape
    scores: dict[tuple[int, ...], float] = {}
    for path in product(range(vocab_size), repeat=frames):
        total = sum(float(log_posteriors[t, path[t]]) for t in range(frames))
        merged: list[int] = []
        for symbol in path:
            if not merged or merged[-1] != symbol:
                merged.append(symbol)
        label = tuple(symbol for symbol in merged if symbol != blank)
        previous = scores.get(label, -math.inf)
        scores[label] = float(np.logaddexp(previous, total))
    return scores


def _random_log_posteriors(seed: int, frames: int, vocab_size: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probs = rng.random((frames, vocab_size)) + 0.01
    probs /= probs.sum(axis=1, keepdims=True)
    return np.log(probs)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_three_frames_three_symbols_matches_brute_force(seed: int) -> None:
    lp = _random_log_posteriors(seed, frames=3, vocab_size=3)
    vocab = Vocab.from_labels(["<blank>", "a", "b"])
    decoder = PrefixBeamSearch(vocab=vocab, beam=64, pruning_threshold=-1000.0)
    nbest = decoder.decode_nbest(lp, 64)
    expected = brute_force_label_scores(lp)
    assert {h.tokens for h in nbest} == set(expected)
    for hyp in nbest:
        want = math.exp(expected[hyp.tokens] / 3)
        assert hyp.confidence == pytest.approx(want, abs=1e-12)
