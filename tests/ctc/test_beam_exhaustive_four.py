"""Exhaustive four-frame agreement with brute-force enumeration."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wavrail.ctc import PrefixBeamSearch
from wavrail.vocab import Vocab

from .test_beam_exhaustive import _random_log_posteriors, brute_force_label_scores


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_four_frames_three_symbols_matches_brute_force(seed: int) -> None:
    lp = _random_log_posteriors(seed, frames=4, vocab_size=3)
    vocab = Vocab.from_labels(["<blank>", "a", "b"])
    decoder = PrefixBeamSearch(vocab=vocab, beam=64, pruning_threshold=-1000.0)
    nbest = decoder.decode_nbest(lp, 64)
    expected = brute_force_label_scores(lp)
    assert {h.tokens for h in nbest} == set(expected)
    for hyp in nbest:
        want = math.exp(expected[hyp.tokens] / 4)
        assert hyp.confidence == pytest.approx(want, abs=1e-12)


def test_golden_fixture_labels_match_brute_force() -> None:
    logits = np.array(
        [
            [2.0, 1.0, 0.0],
            [0.0, 3.0, 0.5],
            [1.0, 1.0, 2.0],
            [0.2, 0.4, 3.0],
        ]
    )
    lp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    vocab = Vocab.from_labels(["<blank>", "a", "b"])
    decoder = PrefixBeamSearch(vocab=vocab, beam=64, pruning_threshold=-1000.0)
    nbest = decoder.decode_nbest(lp, 64)
    expected = brute_force_label_scores(lp)
    for hyp in nbest:
        want = math.exp(expected[hyp.tokens] / 4)
        assert hyp.confidence == pytest.approx(want, abs=1e-12)
