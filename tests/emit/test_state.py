"""Tests for state serialisation and chunk-boundary equivalence."""

from __future__ import annotations

import numpy as np

from wavrail.emit import (
    StreamingDecoder,
    deserialize_state,
    iter_chunks,
    serialize_state,
)
from wavrail.vocab import Vocab


def _vocab() -> Vocab:
    return Vocab.from_labels(["<blank>", "a", "b", "c"])


def test_serialize_roundtrip_preserves_emissions() -> None:
    posteriors = np.array(
        [
            [-5.0, 0.0, -5.0, -5.0],
            [-5.0, 0.0, -5.0, -5.0],
            [0.0, -5.0, -5.0, -5.0],
            [-5.0, -5.0, 0.0, -5.0],
            [-5.0, -5.0, 0.0, -5.0],
            [-5.0, -5.0, -5.0, 0.0],
            [-5.0, -5.0, -1.0, -1.0],
            [0.0, -5.0, -5.0, -5.0],
        ],
        dtype=np.float64,
    )
    decoder = StreamingDecoder(vocab=_vocab(), emission_delay=2)
    state = decoder.initial_state()
    decoder.feed(state, posteriors)
    snapshot = serialize_state(state)
    decoder2 = StreamingDecoder(vocab=_vocab(), emission_delay=2)
    state2 = deserialize_state(snapshot)
    full = decoder.to_hypothesis(state)
    resumed = decoder2.to_hypothesis(state2)
    assert full.tokens == resumed.tokens


def test_chunk_equivalence_with_offline() -> None:
    posteriors = np.array(
        [
            [0.0, -5.0, -5.0, -5.0],
            [-5.0, 0.0, -5.0, -5.0],
            [-5.0, 0.0, -5.0, -5.0],
            [0.0, -5.0, -5.0, -5.0],
            [-5.0, -5.0, 0.0, -5.0],
            [-5.0, -5.0, 0.0, -5.0],
            [0.0, -5.0, -5.0, -5.0],
        ],
        dtype=np.float64,
    )
    decoder = StreamingDecoder(vocab=_vocab(), emission_delay=2)
    state = decoder.initial_state()
    for chunk in iter_chunks(posteriors, 2):
        decoder.feed(state, chunk)
    decoder.finalize(state)
    chunked = decoder.to_hypothesis(state)
    decoder2 = StreamingDecoder(vocab=_vocab(), emission_delay=2)
    state2 = decoder2.initial_state()
    decoder2.feed(state2, posteriors)
    decoder2.finalize(state2)
    flat = decoder2.to_hypothesis(state2)
    assert chunked.tokens == flat.tokens
