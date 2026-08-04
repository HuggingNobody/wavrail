"""Streaming, frame-synchronous emission decoder.

Implements a transducer-style greedy emission loop that walks
the posterior matrix frame by frame. A candidate symbol is only
emitted after it has been the argmax for ``emission_delay``
consecutive frames (default ``1`` which collapses repeats just
like the standard CTC rule).

The state of a streaming decoder is fully serialisable so a
client can pause a real-time stream and resume it later.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from wavrail.ctc._common import Hypothesis
from wavrail.errors import DecodingError
from wavrail.vocab import Vocab


@dataclass
class StreamingDecoderState:
    """Mutable, serialisable state of a :class:`StreamingDecoder`.

    The state records every input frame consumed, the
    accumulated hypothesis tokens, the running per-frame log
    posterior of the chosen symbol, and the candidate symbol with
    its stable-frame count.
    """

    vocab_size: int
    blank_index: int
    emission_delay: int
    max_symbols_per_step: int
    tokens: list[int] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)
    log_probabilities: list[float] = field(default_factory=list)
    candidate_symbol: int = -1
    candidate_count: int = 0
    candidate_frame: int = 0
    frames_consumed: int = 0
    pending_emissions: int = 0


@dataclass
class StreamingDecoder:
    """Frame-synchronous greedy emission decoder.

    Parameters
    ----------
    vocab:
        Token vocabulary. ``vocab.blank_index`` is the id used
        for the silent/blank token.
    emission_delay:
        Number of consecutive frames a candidate symbol must
        remain the argmax before it is emitted. ``1`` matches the
        standard CTC behaviour (consecutive duplicates collapse).
        Larger values trade latency for stability.
    max_symbols_per_step:
        Upper bound on emissions produced by a single ``feed``
        call. Guards against pathological bursts.
    """

    vocab: Vocab
    emission_delay: int = 1
    max_symbols_per_step: int = 16

    def __post_init__(self) -> None:
        if self.emission_delay < 1:
            raise DecodingError("emission_delay must be >= 1")
        if self.max_symbols_per_step < 1:
            raise DecodingError("max_symbols_per_step must be >= 1")

    def initial_state(self) -> StreamingDecoderState:
        return StreamingDecoderState(
            vocab_size=self.vocab.size,
            blank_index=self.vocab.blank_index,
            emission_delay=self.emission_delay,
            max_symbols_per_step=self.max_symbols_per_step,
        )

    def feed(
        self,
        state: StreamingDecoderState,
        frames: np.ndarray,
    ) -> list[int]:
        """Consume ``frames`` and return any newly emitted tokens."""
        if frames.ndim != 2:
            raise DecodingError("frames must be a 2-D array")
        if frames.shape[1] != self.vocab.size:
            raise DecodingError("frames width does not match vocab size")
        emissions: list[int] = []
        for row in frames:
            self._consume_frame(state, row, emissions)
            if len(emissions) >= self.max_symbols_per_step:
                break
        return emissions

    def finalize(self, state: StreamingDecoderState) -> list[int]:
        """Flush any pending candidate at end of stream."""
        candidate = state.candidate_symbol
        last_token = state.tokens[-1] if state.tokens else -1
        if candidate >= 0 and candidate != state.blank_index and last_token != candidate:
            state.tokens.append(candidate)
            state.frame_indices.append(state.candidate_frame)
        state.candidate_symbol = -1
        state.candidate_count = 0
        return []

    def to_hypothesis(self, state: StreamingDecoderState) -> Hypothesis:
        log_p = state.log_probabilities
        confidence = 0.0 if not log_p else math.exp(sum(log_p) / len(log_p))
        text = "".join(self.vocab.id_to_token(t) for t in state.tokens)
        return Hypothesis(
            tokens=tuple(state.tokens),
            text=text,
            confidence=confidence,
            frame_indices=tuple(state.frame_indices),
        )

    def _consume_frame(
        self,
        state: StreamingDecoderState,
        frame: np.ndarray,
        emissions: list[int],
    ) -> None:
        state.frames_consumed += 1
        symbol = int(np.argmax(frame))
        if symbol == state.candidate_symbol:
            state.candidate_count += 1
        else:
            state.candidate_symbol = symbol
            state.candidate_count = 1
            state.candidate_frame = state.frames_consumed - 1
        if state.candidate_count >= state.emission_delay:
            if symbol == state.blank_index:
                state.candidate_count = 0
                state.candidate_symbol = -1
                return
            if not state.tokens or state.tokens[-1] != symbol:
                state.tokens.append(symbol)
                state.frame_indices.append(state.candidate_frame)
                state.log_probabilities.append(float(frame[symbol]))
                emissions.append(symbol)
            state.candidate_count = 0


def decode_streaming(
    log_posteriors: np.ndarray,
    vocab: Vocab,
    emission_delay: int = 1,
) -> Hypothesis:
    """Convenience wrapper that decodes an entire matrix at once."""
    decoder = StreamingDecoder(vocab=vocab, emission_delay=emission_delay)
    state = decoder.initial_state()
    decoder.feed(state, log_posteriors)
    decoder.finalize(state)
    return decoder.to_hypothesis(state)


def serialize_state(state: StreamingDecoderState) -> dict[str, object]:
    """Return a JSON-friendly snapshot of ``state``."""
    return {
        "vocab_size": state.vocab_size,
        "blank_index": state.blank_index,
        "emission_delay": state.emission_delay,
        "max_symbols_per_step": state.max_symbols_per_step,
        "tokens": list(state.tokens),
        "frame_indices": list(state.frame_indices),
        "log_probabilities": list(state.log_probabilities),
        "candidate_symbol": state.candidate_symbol,
        "candidate_count": state.candidate_count,
        "candidate_frame": state.candidate_frame,
        "frames_consumed": state.frames_consumed,
    }


def deserialize_state(payload: dict[str, object]) -> StreamingDecoderState:
    """Restore a :class:`StreamingDecoderState` from ``payload``."""
    return StreamingDecoderState(
        vocab_size=int(cast(Any, payload["vocab_size"])),
        blank_index=int(cast(Any, payload["blank_index"])),
        emission_delay=int(cast(Any, payload["emission_delay"])),
        max_symbols_per_step=int(cast(Any, payload["max_symbols_per_step"])),
        tokens=list(cast(Any, payload.get("tokens", []))),
        frame_indices=list(cast(Any, payload.get("frame_indices", []))),
        log_probabilities=list(cast(Any, payload.get("log_probabilities", []))),
        candidate_symbol=int(cast(Any, payload.get("candidate_symbol", -1))),
        candidate_count=int(cast(Any, payload.get("candidate_count", 0))),
        candidate_frame=int(cast(Any, payload.get("candidate_frame", 0))),
        frames_consumed=int(cast(Any, payload.get("frames_consumed", 0))),
    )


def iter_chunks(
    frames: np.ndarray,
    chunk_size: int,
) -> Iterable[np.ndarray]:
    """Yield successive ``chunk_size`` row slices of ``frames``."""
    if chunk_size < 1:
        raise DecodingError("chunk_size must be >= 1")
    for start in range(0, frames.shape[0], chunk_size):
        yield frames[start : start + chunk_size]


__all__ = [
    "StreamingDecoder",
    "StreamingDecoderState",
    "decode_streaming",
    "deserialize_state",
    "iter_chunks",
    "serialize_state",
]
