"""Streaming feature-and-decode pipeline.

A :class:`StreamingPipeline` owns a fixed-capacity
:class:`RingBuffer`, a feature extractor, and an emission
decoder. The ``feed_samples`` API ingests raw audio chunks and
returns any newly emitted tokens.

The pipeline is fully serialisable so a long-running real-time
process can checkpoint its state and resume across restarts.

Latency is split into two components:

- **algorithmic latency**: the minimum number of frames a
  decoder needs to see before producing a token (controlled by
  the frame size and any lookahead the feature extractor
  requires).
- **emission latency**: the difference in frames between the
  token emission time and the most recent frame consumed.

The :func:`measure_rtf` helper computes a real-time factor by
timing one or more ``feed_samples`` calls and dividing the wall
time by the audio duration processed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from wavrail.ctc._common import Hypothesis
from wavrail.emit.streaming import (
    StreamingDecoder,
    StreamingDecoderState,
)
from wavrail.emit.streaming import (
    deserialize_state as _deserialize_decoder_state,
)
from wavrail.emit.streaming import (
    serialize_state as _serialize_decoder_state,
)
from wavrail.errors import PipelineError
from wavrail.features.frames import frame_signal
from wavrail.stream.buffer import RingBuffer
from wavrail.types import FrameConfig
from wavrail.vocab import Vocab


@dataclass
class StreamingPipelineState:
    """Snapshot of a streaming pipeline's mutable state."""

    buffer_samples: list[float]
    total_samples_consumed: int
    decoder_state: StreamingDecoderState
    feature_lookahead_samples: int
    sample_rate: int
    frame_length_samples: int
    hop_length_samples: int
    cancelled: bool = False


@dataclass
class StreamingPipeline:
    """Glue a ring buffer, feature extractor, and emission decoder."""

    sample_rate: int
    frame_length_ms: float = 25.0
    hop_length_ms: float = 10.0
    pre_emphasis: float = 0.0
    feature_dim: int = 13
    emission_delay: int = 1
    vocab: Vocab | None = None
    decoder: StreamingDecoder | None = None
    buffer_capacity_seconds: float = 5.0
    overflow: str = "error"
    feature_extractor: Callable[[np.ndarray], np.ndarray] | None = None
    state: StreamingPipelineState | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise PipelineError("sample_rate must be > 0")
        if self.frame_length_ms <= 0:
            raise PipelineError("frame_length_ms must be > 0")
        if self.hop_length_ms <= 0:
            raise PipelineError("hop_length_ms must be > 0")
        if self.vocab is None and self.decoder is None:
            raise PipelineError("either vocab or decoder must be provided")
        self.frame_length_samples = max(1, round(self.sample_rate * self.frame_length_ms / 1000.0))
        self.hop_length_samples = max(1, round(self.sample_rate * self.hop_length_ms / 1000.0))
        self.feature_lookahead_samples = self.frame_length_samples - self.hop_length_samples
        self.buffer_capacity_samples = max(
            self.frame_length_samples * 4,
            int(self.sample_rate * self.buffer_capacity_seconds),
        )
        self.buffer = RingBuffer(capacity=self.buffer_capacity_samples, overflow=self.overflow)
        if self.decoder is None:
            assert self.vocab is not None
            self.decoder = StreamingDecoder(vocab=self.vocab, emission_delay=self.emission_delay)
        if self.feature_extractor is None:
            self.feature_extractor = _default_feature_extractor(self.feature_dim)
        # center=False anchors the frame grid at absolute sample zero:
        # the buffer keeps the unframed tail instead of re-padding it on
        # every feed, which is what makes chunked feeding frame-identical
        # to offline blocking.
        self.frame_config = FrameConfig(
            frame_length=self.frame_length_samples,
            hop_length=self.hop_length_samples,
            center=False,
        )
        self.state = StreamingPipelineState(
            buffer_samples=[],
            total_samples_consumed=0,
            decoder_state=self.decoder.initial_state(),
            feature_lookahead_samples=self.feature_lookahead_samples,
            sample_rate=self.sample_rate,
            frame_length_samples=self.frame_length_samples,
            hop_length_samples=self.hop_length_samples,
        )

    @property
    def buffered_samples(self) -> int:
        return self.buffer.size

    @property
    def is_cancelled(self) -> bool:
        return bool(self.state is not None and self.state.cancelled)

    def cancel(self) -> None:
        """Cancel the stream.

        Subsequent :meth:`feed_samples` calls raise until :meth:`reset`
        restarts the pipeline. :meth:`finalize` still works so callers
        can retrieve the hypothesis emitted before cancellation.
        """
        assert self.state is not None
        self.state.cancelled = True

    @property
    def algorithmic_latency_ms(self) -> float:
        """Earliest emission time for the very first audio sample.

        One frame length of accumulation plus ``emission_delay - 1``
        confirmation hops. This is frame-grid accounting, not
        wall-clock scheduling delay.
        """
        assert self.decoder is not None
        delay = max(1, int(self.decoder.emission_delay))
        samples = self.frame_length_samples + (delay - 1) * self.hop_length_samples
        return samples / self.sample_rate * 1000.0

    def feed_samples(self, samples: np.ndarray) -> list[int]:
        """Push a chunk of mono audio through the pipeline."""
        if self.is_cancelled:
            raise PipelineError("pipeline was cancelled; call reset() to restart")
        if samples.ndim != 1:
            raise PipelineError("samples must be 1-D (mono)")
        self.buffer.extend(samples.reshape(-1, 1))
        snapshot = self.buffer.snapshot()
        if snapshot.shape[0] < self.frame_length_samples:
            self._sync_state()
            return []
        frames = frame_signal(snapshot, self.frame_config)
        features = self.feature_extractor(frames)
        emissions = self.decoder.feed(self.state.decoder_state, features)
        consumed = min(self.hop_length_samples * frames.shape[0], snapshot.shape[0])
        self._consume(consumed)
        self.state.total_samples_consumed += int(samples.shape[0])
        self._sync_state()
        return emissions

    def _consume(self, n: int) -> None:
        self.buffer.consume(n)

    def _sync_state(self) -> None:
        """Mirror the buffered tail into the serialisable state."""
        assert self.state is not None
        self.state.buffer_samples = self.buffer.snapshot().reshape(-1).tolist()

    def finalize(self) -> Hypothesis:
        """Flush the pipeline and return the final hypothesis."""
        snapshot = self.buffer.snapshot()
        if snapshot.shape[0] >= self.frame_length_samples:
            frames = frame_signal(snapshot, self.frame_config)
            features = self.feature_extractor(frames)
            self.decoder.feed(self.state.decoder_state, features)
        self.decoder.finalize(self.state.decoder_state)
        self.buffer.reset()
        self._sync_state()
        return self.decoder.to_hypothesis(self.state.decoder_state)

    def load_state(self, state: StreamingPipelineState) -> None:
        """Adopt ``state`` so a paused stream resumes where it stopped.

        Sample rate, frame geometry, and decoder vocabulary must all
        match this pipeline; otherwise resumption would silently shift
        the frame grid or reinterpret token ids.
        """
        if state.sample_rate != self.sample_rate:
            raise PipelineError("sample_rate mismatch; refusing to resume")
        if (
            state.frame_length_samples != self.frame_length_samples
            or state.hop_length_samples != self.hop_length_samples
        ):
            raise PipelineError("frame geometry mismatch; refusing to resume")
        assert self.decoder is not None
        if state.decoder_state.vocab_size != self.decoder.vocab.size:
            raise PipelineError("decoder vocab mismatch; refusing to resume")
        if len(state.buffer_samples) > self.buffer.capacity:
            raise PipelineError("buffered tail exceeds this pipeline's capacity")
        self.buffer.reset()
        if state.buffer_samples:
            self.buffer.extend(np.asarray(state.buffer_samples, dtype=np.float32))
        self.state = state

    def reset(self) -> None:
        self.buffer.reset()
        self.state = StreamingPipelineState(
            buffer_samples=[],
            total_samples_consumed=0,
            decoder_state=self.decoder.initial_state(),
            feature_lookahead_samples=self.feature_lookahead_samples,
            sample_rate=self.sample_rate,
            frame_length_samples=self.frame_length_samples,
            hop_length_samples=self.hop_length_samples,
        )


def _default_feature_extractor(feature_dim: int) -> Callable[[np.ndarray], np.ndarray]:
    def _featurise(frames: np.ndarray) -> np.ndarray:
        if feature_dim <= 0:
            return frames.astype(np.float64)
        mean = frames.mean(axis=1, keepdims=True)
        std = frames.std(axis=1, keepdims=True) + 1e-8
        return ((frames - mean) / std)[:, :feature_dim].astype(np.float64)

    return _featurise


def measure_rtf(
    pipeline: StreamingPipeline,
    samples: np.ndarray,
    *,
    chunk_size: int = 1600,
) -> dict[str, float]:
    """Run ``pipeline`` over ``samples`` and report real-time factor."""
    if samples.ndim != 1:
        raise PipelineError("samples must be 1-D (mono)")
    duration = samples.shape[0] / float(pipeline.sample_rate)
    if duration <= 0:
        return {"duration_seconds": 0.0, "elapsed_seconds": 0.0, "rtf": 0.0}
    pipeline.reset()
    start = time.perf_counter()
    for start_idx in range(0, samples.shape[0], chunk_size):
        pipeline.feed_samples(samples[start_idx : start_idx + chunk_size])
    elapsed = time.perf_counter() - start
    return {
        "duration_seconds": float(duration),
        "elapsed_seconds": float(elapsed),
        "rtf": float(elapsed / duration),
    }


def serialize_pipeline_state(state: StreamingPipelineState) -> dict[str, Any]:
    return {
        "buffer_samples": list(state.buffer_samples),
        "total_samples_consumed": state.total_samples_consumed,
        "decoder_state": _serialize_decoder_state(state.decoder_state),
        "feature_lookahead_samples": state.feature_lookahead_samples,
        "sample_rate": state.sample_rate,
        "frame_length_samples": state.frame_length_samples,
        "hop_length_samples": state.hop_length_samples,
        "cancelled": state.cancelled,
    }


def deserialize_pipeline_state(payload: dict[str, Any]) -> StreamingPipelineState:
    return StreamingPipelineState(
        buffer_samples=list(payload.get("buffer_samples", [])),
        total_samples_consumed=int(payload.get("total_samples_consumed", 0)),
        decoder_state=_deserialize_decoder_state(payload["decoder_state"]),
        feature_lookahead_samples=int(payload.get("feature_lookahead_samples", 0)),
        sample_rate=int(payload.get("sample_rate", 16000)),
        frame_length_samples=int(payload.get("frame_length_samples", 400)),
        hop_length_samples=int(payload.get("hop_length_samples", 160)),
        cancelled=bool(payload.get("cancelled", False)),
    )


__all__ = [
    "StreamingPipeline",
    "StreamingPipelineState",
    "deserialize_pipeline_state",
    "measure_rtf",
    "serialize_pipeline_state",
]
