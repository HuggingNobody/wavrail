"""Push-based frame blocking for streaming pipelines.

:class:`StreamingFramer` accepts arbitrarily sized sample chunks and
emits only complete frames, holding the overlap of the last emitted
frame in a bounded pending buffer. The equivalence contract with the
batch blocker is exact: for ``center=False`` configs, concatenating
every frame a framer emits for a signal equals
``frame_signal(signal, config)``.

``center=True`` is rejected at construction: centred framing
reflect-pads around the *whole* signal, which cannot be produced
incrementally without changing frame values mid-stream.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from wavrail.errors import PipelineError
from wavrail.types import AudioArray, FeatureArray, FrameConfig

FRAMER_STATE_VERSION = 1


class StreamingFramer:
    """Accumulate samples and emit complete overlapping frames.

    After every :meth:`push` the pending buffer holds fewer than
    ``frame_length`` samples, so memory use is bounded regardless of
    how much audio the stream delivers.
    """

    def __init__(self, config: FrameConfig) -> None:
        if config.center:
            raise PipelineError(
                "StreamingFramer requires center=False; centred padding needs the whole signal"
            )
        self._config = config
        self._pending = np.zeros(0, dtype=np.float32)
        self._total_in = 0
        self._total_frames = 0

    @property
    def config(self) -> FrameConfig:
        """The frame configuration the framer was built with."""
        return self._config

    @property
    def pending_samples(self) -> int:
        """Number of buffered samples not yet emitted as frames."""
        return int(self._pending.size)

    @property
    def total_input_samples(self) -> int:
        """Total samples pushed over the framer's lifetime."""
        return self._total_in

    @property
    def total_frames_emitted(self) -> int:
        """Total frames emitted over the framer's lifetime."""
        return self._total_frames

    def pending_array(self) -> np.ndarray:
        """Return a copy of the pending sample buffer."""
        return self._pending.copy()

    def restore(
        self,
        pending: np.ndarray,
        total_input_samples: int,
        total_frames_emitted: int,
    ) -> None:
        """Reinstall serialised state; used by :func:`deserialize_framer_state`."""
        arr = np.asarray(pending, dtype=np.float32).reshape(-1)
        if arr.size >= self._config.frame_length:
            raise PipelineError("pending buffer must be shorter than frame_length")
        if total_input_samples < arr.size or total_frames_emitted < 0:
            raise PipelineError("inconsistent framer counters")
        self._pending = arr
        self._total_in = int(total_input_samples)
        self._total_frames = int(total_frames_emitted)

    def push(self, samples: AudioArray) -> FeatureArray:
        """Feed one chunk; return the newly completed frames.

        The result has shape ``(num_new_frames, frame_length)`` and is
        ``float32``; an empty chunk (or one too small to complete a
        frame) yields a zero-row array.
        """
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        if arr.size:
            self._pending = np.concatenate([self._pending, arr])
            self._total_in += arr.size
        return self._drain()

    def flush(self, pad_tail: bool = False) -> FeatureArray:
        """Release the tail under the chosen policy and clear pending.

        ``pad_tail=False`` (the default) drops the remaining
        sub-frame samples, which is what :func:`frame_signal` does.
        ``pad_tail=True`` emits one final frame zero-padded to
        ``frame_length`` when at least one sample is pending, so no
        audio is silently discarded at end of stream.
        """
        cfg = self._config
        tail = self._pending
        self._pending = np.zeros(0, dtype=np.float32)
        if not pad_tail or tail.size == 0:
            return np.zeros((0, cfg.frame_length), dtype=np.float32)
        frame = np.zeros(cfg.frame_length, dtype=np.float32)
        frame[: tail.size] = tail
        self._total_frames += 1
        return frame.reshape(1, -1)

    @property
    def lookahead_samples(self) -> int:
        """Overlap retained between consecutive emitted frames."""
        return self._config.frame_length - self._config.hop_length

    @property
    def max_buffered_samples(self) -> int:
        """Upper bound on pending samples after any completed push."""
        return self._config.frame_length - 1

    def emission_latency_ms(self, sample_rate: int) -> float:
        """Milliseconds between a frame's start and its emission.

        A frame covering ``[start, start + frame_length)`` can only be
        emitted once its last sample has arrived, so the algorithmic
        emission latency is exactly one frame length.
        """
        if sample_rate <= 0:
            raise PipelineError("sample_rate must be positive")
        return self._config.frame_length / sample_rate * 1000.0

    def _drain(self) -> FeatureArray:
        cfg = self._config
        available = self._pending.size
        if available < cfg.frame_length:
            return np.zeros((0, cfg.frame_length), dtype=np.float32)
        count = (available - cfg.frame_length) // cfg.hop_length + 1
        indices = np.arange(cfg.frame_length)[None, :] + cfg.hop_length * np.arange(count)[:, None]
        frames = self._pending[indices].astype(np.float32)
        consumed = count * cfg.hop_length
        self._pending = self._pending[consumed:]
        self._total_frames += count
        return frames


def serialize_framer_state(framer: StreamingFramer) -> dict[str, object]:
    """Return a JSON-ready snapshot of ``framer``.

    The payload carries the schema version, the full frame config,
    the lifetime counters, and the pending buffer so a paused stream
    can resume elsewhere with bit-identical framing.
    """
    cfg = framer.config
    return {
        "schema_version": FRAMER_STATE_VERSION,
        "frame_length": cfg.frame_length,
        "hop_length": cfg.hop_length,
        "pad_mode": cfg.pad_mode,
        "center": cfg.center,
        "total_input_samples": framer.total_input_samples,
        "total_frames_emitted": framer.total_frames_emitted,
        "pending": [float(x) for x in framer.pending_array()],
    }


def deserialize_framer_state(payload: dict[str, object]) -> StreamingFramer:
    """Rebuild a framer from :func:`serialize_framer_state` output."""
    version = payload.get("schema_version")
    if version != FRAMER_STATE_VERSION:
        raise PipelineError(
            f"incompatible schema_version={version!r}; expected {FRAMER_STATE_VERSION!r}"
        )
    config = FrameConfig(
        frame_length=int(cast(Any, payload["frame_length"])),
        hop_length=int(cast(Any, payload["hop_length"])),
        pad_mode=str(payload["pad_mode"]),
        center=bool(payload["center"]),
    )
    framer = StreamingFramer(config)
    framer.restore(
        pending=np.asarray(cast(list, payload["pending"]), dtype=np.float32),
        total_input_samples=int(cast(Any, payload["total_input_samples"])),
        total_frames_emitted=int(cast(Any, payload["total_frames_emitted"])),
    )
    return framer


__all__ = [
    "FRAMER_STATE_VERSION",
    "StreamingFramer",
    "deserialize_framer_state",
    "serialize_framer_state",
]
