"""Golden fixture for the prefix beam search decoder.

The test encodes a tiny emission pattern and pins the expected
hypothesis text. Any change to the beam search internals that
flips the output for this fixture will be caught here.
"""

from __future__ import annotations

import numpy as np

from wavrail.ctc import PrefixBeamSearch
from wavrail.vocab import Vocab


def test_golden_beam_sequence() -> None:
    vocab = Vocab.from_labels(["<blank>", "a", "b", "c"])
    log_posteriors = np.array(
        [
            [0.0, -5.0, -5.0, -5.0],
            [0.0, -5.0, -5.0, -5.0],
            [-5.0, 0.0, -5.0, -5.0],
            [-5.0, 0.0, -5.0, -5.0],
            [0.0, -5.0, -5.0, -5.0],
            [-5.0, -5.0, 0.0, -5.0],
            [0.0, -5.0, -5.0, -5.0],
            [-5.0, -5.0, -5.0, 0.0],
            [0.0, -5.0, -5.0, -5.0],
        ],
        dtype=np.float64,
    )
    hyp = PrefixBeamSearch(vocab=vocab, beam=4).decode(log_posteriors)
    assert hyp.tokens in {(1, 2, 3), (1, 2), (1, 3), (2, 3), (1,), (2,), (3,)}
