"""Greedy decoding is invariant under temperature rescaling.

Temperature changes posterior sharpness but never the row argmax, and
greedy decoding depends only on the argmax sequence — so the decoded
text must be identical at every positive temperature.
"""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.ctc import greedy_decode, rescale_temperature
from wavrail.vocab import Vocab


@pytest.mark.parametrize("temperature", [0.25, 0.5, 2.0, 8.0])
def test_greedy_text_stable_under_temperature(temperature: float) -> None:
    vocab = Vocab.from_labels(["<blank>", "a", "b", "c"])
    rng = np.random.default_rng(3)
    rows = rng.random((30, 4)) + 0.01
    posteriors = rows / rows.sum(axis=1, keepdims=True)

    base = greedy_decode(posteriors, vocab)
    rescaled = np.exp(rescale_temperature(np.log(posteriors), temperature))
    hyp = greedy_decode(rescaled, vocab)
    assert hyp.tokens == base.tokens
    assert hyp.text == base.text
