"""Behaviour tests for the streaming emission decoder."""

from __future__ import annotations

import numpy as np

from wavrail.emit import StreamingDecoder, decode_streaming
from wavrail.vocab import Vocab


def _vocab() -> Vocab:
    return Vocab.from_labels(["<blank>", "a", "b", "c"])


def test_emits_only_after_repeat_window() -> None:
    posteriors = np.array(
        [
            [0.0, -5.0, -5.0, -5.0],
            [0.0, -5.0, -5.0, -5.0],
            [-5.0, 0.0, -5.0, -5.0],
            [-5.0, 0.0, -5.0, -5.0],
            [-5.0, -5.0, 0.0, -5.0],
            [-5.0, -5.0, 0.0, -5.0],
        ],
        dtype=np.float64,
    )
    hyp = decode_streaming(posteriors, _vocab(), emission_delay=2)
    assert hyp.tokens == (1, 2)


def test_blank_breaks_run() -> None:
    posteriors = np.array(
        [
            [-5.0, 0.0, -5.0, -5.0],
            [0.0, -5.0, -5.0, -5.0],
            [-5.0, 0.0, -5.0, -5.0],
        ],
        dtype=np.float64,
    )
    hyp = decode_streaming(posteriors, _vocab(), emission_delay=2)
    assert hyp.tokens == (1,)


def test_emit_different_after_blank() -> None:
    posteriors = np.array(
        [
            [-5.0, 0.0, -5.0, -5.0],
            [-5.0, 0.0, -5.0, -5.0],
            [0.0, -5.0, -5.0, -5.0],
            [-5.0, -5.0, 0.0, -5.0],
            [-5.0, -5.0, 0.0, -5.0],
        ],
        dtype=np.float64,
    )
    hyp = decode_streaming(posteriors, _vocab(), emission_delay=2)
    assert hyp.tokens == (1, 2)


def test_state_resume_roundtrip() -> None:
    posteriors = np.array(
        [
            [-5.0, 0.0, -5.0, -5.0],
            [-5.0, 0.0, -5.0, -5.0],
            [-5.0, -5.0, 0.0, -5.0],
            [-5.0, -5.0, 0.0, -5.0],
            [-5.0, 0.0, -5.0, -5.0],
        ],
        dtype=np.float64,
    )
    decoder = StreamingDecoder(vocab=_vocab(), emission_delay=2)
    state = decoder.initial_state()
    first_half = decoder.feed(state, posteriors[:2])
    # Restart from a fresh decoder that has already consumed the same prefix.
    decoder2 = StreamingDecoder(vocab=_vocab(), emission_delay=2)
    state2 = decoder2.initial_state()
    decoder2.feed(state2, posteriors[:2])
    second_half = decoder.feed(state, posteriors[2:])
    second_half_resumed = decoder2.feed(state2, posteriors[2:])
    assert second_half == second_half_resumed
    assert first_half == [1]
