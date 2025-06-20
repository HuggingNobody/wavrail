"""N-best list semantics of the prefix beam decoder."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.ctc import PrefixBeamSearch
from wavrail.errors import DecodingError
from wavrail.vocab import Vocab


def _vocab() -> Vocab:
    return Vocab.from_labels(["<blank>", "a", "b"])


def _random_log_posteriors(seed: int, frames: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probs = rng.random((frames, 3)) + 0.05
    probs /= probs.sum(axis=1, keepdims=True)
    return np.log(probs)


def test_nbest_is_sorted_and_unique() -> None:
    lp = _random_log_posteriors(4)
    decoder = PrefixBeamSearch(vocab=_vocab(), beam=16, pruning_threshold=-1000.0)
    nbest = decoder.decode_nbest(lp, 8)
    confidences = [h.confidence for h in nbest]
    assert confidences == sorted(confidences, reverse=True)
    tokens = [h.tokens for h in nbest]
    assert len(set(tokens)) == len(tokens)


def test_nbest_top_matches_decode() -> None:
    lp = _random_log_posteriors(5)
    decoder = PrefixBeamSearch(vocab=_vocab(), beam=8)
    assert decoder.decode_nbest(lp, 3)[0].tokens == decoder.decode(lp).tokens


def test_single_frame_lists_every_reachable_prefix() -> None:
    lp = np.array([[-9.0, -0.5, -5.0]])
    decoder = PrefixBeamSearch(vocab=_vocab(), beam=16)
    nbest = decoder.decode_nbest(lp, 10)
    assert {h.tokens for h in nbest} == {(), (1,), (2,)}


def test_nbest_rejects_k_below_one() -> None:
    decoder = PrefixBeamSearch(vocab=_vocab(), beam=4)
    with pytest.raises(DecodingError, match="k"):
        decoder.decode_nbest(_random_log_posteriors(6), 0)


def test_nbest_empty_input_returns_single_empty() -> None:
    decoder = PrefixBeamSearch(vocab=_vocab(), beam=4)
    nbest = decoder.decode_nbest(np.zeros((0, 3), dtype=np.float64), 5)
    assert len(nbest) == 1
    assert nbest[0].tokens == ()
