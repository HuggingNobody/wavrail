"""Degenerate-input edges for the prefix beam decoder."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wavrail.ctc import PrefixBeamSearch
from wavrail.vocab import Vocab


def _vocab3() -> Vocab:
    return Vocab.from_labels(["<blank>", "a", "b"])


def test_blank_only_vocab_returns_empty() -> None:
    vocab = Vocab.from_labels(["<blank>"])
    hyp = PrefixBeamSearch(vocab=vocab, beam=2).decode(np.zeros((3, 1)))
    assert hyp.tokens == ()
    assert hyp.text == ""


def test_single_frame_picks_best_symbol() -> None:
    lp = np.array([[-9.0, -0.1, -5.0]])
    hyp = PrefixBeamSearch(vocab=_vocab3(), beam=3).decode(lp)
    assert hyp.tokens == (1,)
    assert hyp.confidence == pytest.approx(math.exp(-0.1), abs=1e-9)


def test_all_neginf_frame_yields_empty_hypothesis() -> None:
    lp = np.full((2, 3), -np.inf)
    hyp = PrefixBeamSearch(vocab=_vocab3(), beam=3).decode(lp)
    assert hyp.tokens == ()
    assert hyp.confidence == 0.0


def test_decode_is_deterministic() -> None:
    rng = np.random.default_rng(21)
    probs = rng.random((5, 3)) + 0.01
    probs /= probs.sum(axis=1, keepdims=True)
    lp = np.log(probs)
    decoder = PrefixBeamSearch(vocab=_vocab3(), beam=6)
    first = decoder.decode(lp)
    second = decoder.decode(lp)
    assert first.tokens == second.tokens
    assert first.confidence == second.confidence
