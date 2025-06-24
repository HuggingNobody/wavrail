"""A wide, unpruned beam must contain the greedy label sequence.

Inputs are near-peaked random distributions (exponent 4 on uniform
draws), which keep the greedy path in the high-probability region
where the containment property is meaningful for moderate beams.
"""

from __future__ import annotations

import numpy as np

from wavrail.ctc import PrefixBeamSearch, greedy_decode
from wavrail.vocab import Vocab


def test_wide_beam_contains_greedy() -> None:
    vocab = Vocab.from_labels(["<blank>", "a", "b", "c"])
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        draws = rng.random((6, 4)) ** 4 * 3 + 0.01
        probs = draws / draws.sum(axis=1, keepdims=True)
        lp = np.log(probs)
        greedy = greedy_decode(lp, vocab).tokens
        decoder = PrefixBeamSearch(vocab=vocab, beam=100, pruning_threshold=-1000.0)
        nbest = decoder.decode_nbest(lp, 100)
        assert greedy in {h.tokens for h in nbest}
