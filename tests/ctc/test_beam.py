"""Tests for the prefix beam search decoder."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.ctc import PrefixBeamSearch
from wavrail.errors import DecodingError
from wavrail.vocab import Vocab


def _vocab() -> Vocab:
    return Vocab.from_labels(["<blank>", "a", "b", "c"])


def _log_posterior(sequence: list[int], num_classes: int = 4) -> np.ndarray:
    arr = np.full((len(sequence), num_classes), -10.0, dtype=np.float64)
    for t, idx in enumerate(sequence):
        arr[t, idx] = 0.0
    return arr


def test_beam_runs_without_error() -> None:
    log_posteriors = _log_posterior([0, 1, 0, 2, 3, 0])
    beam = PrefixBeamSearch(vocab=_vocab(), beam=4).decode(log_posteriors)
    assert isinstance(beam.tokens, tuple)
    assert -0.001 <= beam.confidence <= 1.001


def test_beam_produces_hypothesis_for_short_input() -> None:
    log_posteriors = _log_posterior([0, 1, 0])
    beam = PrefixBeamSearch(vocab=_vocab(), beam=3).decode(log_posteriors)
    assert isinstance(beam.tokens, tuple)
    assert isinstance(beam.text, str)


def test_beam_rejects_zero_beam() -> None:
    with pytest.raises(DecodingError):
        PrefixBeamSearch(vocab=_vocab(), beam=0)


def test_beam_rejects_shape_mismatch() -> None:
    log_posteriors = np.zeros((4, 6), dtype=np.float64)
    decoder = PrefixBeamSearch(vocab=_vocab(), beam=3)
    with pytest.raises(DecodingError):
        decoder.decode(log_posteriors)


def test_beam_rejects_3d_input() -> None:
    log_posteriors = np.zeros((3, 4, 5), dtype=np.float64)
    decoder = PrefixBeamSearch(vocab=_vocab(), beam=3)
    with pytest.raises(DecodingError):
        decoder.decode(log_posteriors)


def test_beam_empty_input() -> None:
    decoder = PrefixBeamSearch(vocab=_vocab(), beam=3)
    log_posteriors = np.zeros((0, 4), dtype=np.float64)
    hyp = decoder.decode(log_posteriors)
    assert hyp.tokens == ()
    assert hyp.text == ""
