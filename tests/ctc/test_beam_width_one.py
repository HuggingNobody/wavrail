"""With a beam of one, prefix search must reproduce greedy decoding."""

from __future__ import annotations

import numpy as np

from wavrail.ctc import PrefixBeamSearch, greedy_decode
from wavrail.vocab import Vocab


def test_beam_one_equals_greedy_on_peaked_inputs() -> None:
    vocab = Vocab.from_labels(["<blank>", "a", "b", "c"])
    rng = np.random.default_rng(9)
    for _ in range(50):
        frames = int(rng.integers(3, 9))
        sequence = rng.integers(0, 4, frames)
        lp = np.full((frames, 4), -12.0)
        for t, symbol in enumerate(sequence):
            lp[t, symbol] = 0.0
        beam = PrefixBeamSearch(vocab=vocab, beam=1).decode(lp)
        assert beam.tokens == greedy_decode(lp, vocab).tokens
