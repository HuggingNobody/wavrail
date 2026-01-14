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

import numpy as np

from wavrail.errors import PipelineError
from wavrail.types import AudioArray, FeatureArray, FrameConfig


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

    def _drain(self) -> FeatureArray:
        cfg = self._config
        available = self._pending.size
        if available < cfg.frame_length:
            return np.zeros((0, cfg.frame_length), dtype=np.float32)
        count = (available - cfg.frame_length) // cfg.hop_length + 1
        indices = np.arange(cfg.frame_length)[None, :] + cfg.hop_length * np.arange(count)[:, None]
        frames = self._pending[indices].astype(np.float32)
        consumed = (count - 1) * cfg.hop_length + cfg.frame_length
        self._pending = self._pending[consumed:]
        self._total_frames += count
        return frames


__all__ = ["StreamingFramer"]
