"""Streaming power-spectrogram adapter.

:class:`StreamingSpectrogram` pairs a :class:`StreamingFramer` with
:func:`wavrail.features.power_spectrum` so chunked audio can be
turned into spectrum rows incrementally. Because both halves are the
same code paths the batch pipeline uses, the rows a stream produces
are exactly the rows ``power_spectrum(frame_signal(...))`` produces
for the same signal — the tests pin that equivalence.
"""

from __future__ import annotations

import numpy as np

from wavrail.errors import PipelineError
from wavrail.features.spectrum import power_spectrum
from wavrail.stream.framer import StreamingFramer
from wavrail.types import AudioArray, FeatureArray, FrameConfig


class StreamingSpectrogram:
    """Push-based power spectrogram with offline equivalence."""

    def __init__(
        self,
        frame_config: FrameConfig,
        fft_size: int,
        window: np.ndarray | None = None,
    ) -> None:
        if fft_size <= 0:
            raise PipelineError("fft_size must be positive")
        if fft_size < frame_config.frame_length:
            raise PipelineError("fft_size must be >= frame_length")
        self._framer = StreamingFramer(frame_config)
        self._fft_size = fft_size
        self._window = window

    @property
    def num_bins(self) -> int:
        """Number of spectrum bins per row (``fft_size // 2 + 1``)."""
        return self._fft_size // 2 + 1

    @property
    def framer(self) -> StreamingFramer:
        """The underlying framer, for state inspection and serde."""
        return self._framer

    def push(self, samples: AudioArray) -> FeatureArray:
        """Feed one chunk; return spectrum rows for completed frames."""
        return self._spectrum(self._framer.push(samples))

    def flush(self, pad_tail: bool = False) -> FeatureArray:
        """Flush the framer tail and return its spectrum rows."""
        return self._spectrum(self._framer.flush(pad_tail=pad_tail))

    def _spectrum(self, frames: FeatureArray) -> FeatureArray:
        if frames.shape[0] == 0:
            return np.zeros((0, self.num_bins), dtype=np.float32)
        return power_spectrum(frames, fft_size=self._fft_size, window=self._window)


__all__ = ["StreamingSpectrogram"]
