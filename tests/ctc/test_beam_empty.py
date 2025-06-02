"""Beam decoder edge cases."""

from __future__ import annotations

import numpy as np

from wavrail.ctc import PrefixBeamSearch
from wavrail.vocab import Vocab


def test_empty_input() -> None:
    vocab = Vocab.from_labels(["<blank>", "a", "b"])
    hyp = PrefixBeamSearch(vocab=vocab, beam=3).decode(np.zeros((0, 3), dtype=np.float64))
    assert hyp.tokens == ()


def test_beam_includes_greedy_result() -> None:
    vocab = Vocab.from_labels(["<blank>", "a", "b", "c"])
    posteriors = np.array(
        [
            [-5.0, 0.0, -5.0, -5.0],
            [0.0, -5.0, -5.0, -5.0],
            [-5.0, 0.0, -5.0, -5.0],
            [-5.0, -5.0, 0.0, -5.0],
        ],
        dtype=np.float64,
    )
    beam = PrefixBeamSearch(vocab=vocab, beam=2).decode(posteriors)
    assert beam.tokens == (1, 2)
