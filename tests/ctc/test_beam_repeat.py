"""The beam must represent repeats separated by blanks."""

from __future__ import annotations

import numpy as np

from wavrail.ctc import PrefixBeamSearch, greedy_decode
from wavrail.vocab import Vocab


def _vocab() -> Vocab:
    return Vocab.from_labels(["<blank>", "a", "b"])


def _peaked(sequence: list[int]) -> np.ndarray:
    arr = np.full((len(sequence), 3), -10.0)
    for t, idx in enumerate(sequence):
        arr[t, idx] = 0.0
    return arr


def test_repeat_across_blank_extends() -> None:
    decoder = PrefixBeamSearch(vocab=_vocab(), beam=4)
    hyp = decoder.decode(_peaked([1, 0, 1]))
    assert hyp.tokens == (1, 1)
    assert hyp.text == "aa"


def test_repeat_without_blank_collapses() -> None:
    decoder = PrefixBeamSearch(vocab=_vocab(), beam=4)
    hyp = decoder.decode(_peaked([1, 1, 1]))
    assert hyp.tokens == (1,)


def test_alternating_blank_pattern_matches_greedy() -> None:
    lp = _peaked([1, 0, 2, 0, 1])
    decoder = PrefixBeamSearch(vocab=_vocab(), beam=4)
    hyp = decoder.decode(lp)
    assert hyp.tokens == (1, 2, 1)
    assert hyp.tokens == greedy_decode(lp, _vocab()).tokens


def test_all_blanks_decodes_to_empty() -> None:
    decoder = PrefixBeamSearch(vocab=_vocab(), beam=4)
    hyp = decoder.decode(_peaked([0, 0, 0]))
    assert hyp.tokens == ()
    assert hyp.confidence > 0.99
