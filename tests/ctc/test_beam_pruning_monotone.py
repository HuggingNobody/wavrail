"""Monotone pruning: looser thresholds never lose score mass."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.ctc import PrefixBeamSearch
from wavrail.vocab import Vocab


def _vocab() -> Vocab:
    return Vocab.from_labels(["<blank>", "a", "b", "c"])


@pytest.mark.parametrize("beam", [2, 4, 8])
def test_looser_threshold_never_worsens_best(beam: int) -> None:
    for seed in range(6):
        rng = np.random.default_rng(seed)
        probs = rng.random((7, 4)) + 0.01
        probs /= probs.sum(axis=1, keepdims=True)
        lp = np.log(probs)
        tight = PrefixBeamSearch(vocab=_vocab(), beam=beam, pruning_threshold=-0.5).decode(lp)
        loose = PrefixBeamSearch(vocab=_vocab(), beam=beam, pruning_threshold=-25.0).decode(lp)
        assert loose.confidence >= tight.confidence - 1e-12
