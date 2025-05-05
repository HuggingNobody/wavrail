"""Bounded ring buffer for streaming audio frames.

The buffer is backed by a NumPy array of fixed capacity. Old
samples are overwritten when the buffer wraps around. The
buffer never grows unboundedly; pushing more samples than the
capacity allows raises an explicit ``BufferError`` so callers
can decide to flush, drop, or terminate.
"""

from __future__ import annotations

import numpy as np

from wavrail.errors import PipelineError as _BufferError


class RingBuffer:
    """Fixed-capacity NumPy-backed ring buffer."""

    def __init__(self, capacity: int, channels: int = 1, dtype: np.dtype = np.float32) -> None:
        if capacity < 1:
            raise _BufferError("capacity must be >= 1")
        if channels < 1:
            raise _BufferError("channels must be >= 1")
        self.capacity = int(capacity)
        self.channels = int(channels)
        self.dtype = np.dtype(dtype)
        self._buffer = np.zeros((self.capacity, self.channels), dtype=self.dtype)
        self._size = 0
        self._head = 0  # next write index

    @property
    def size(self) -> int:
        return self._size

    @property
    def total_written(self) -> int:
        return self._head

    def extend(self, samples: np.ndarray) -> None:
        """Append ``samples`` of shape ``(N, channels)`` to the buffer."""
        if samples.ndim == 1:
            samples = samples.reshape(-1, self.channels)
        if samples.ndim != 2 or samples.shape[1] != self.channels:
            raise _BufferError(f"samples must have shape (N, {self.channels})")
        n = samples.shape[0]
        if n == 0:
            return
        if n > self.capacity:
            raise _BufferError("cannot push more samples than capacity")
        end = self._head + n
        if end <= self.capacity:
            self._buffer[self._head : end] = samples
        else:
            first = self.capacity - self._head
            self._buffer[self._head :] = samples[:first]
            self._buffer[: n - first] = samples[first:]
        self._head = (self._head + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def drain(self) -> np.ndarray:
        """Return the buffered samples in chronological order and empty the buffer."""
        if self._size == 0:
            return np.zeros((0, self.channels), dtype=self.dtype)
        if self._head >= self._size:
            start = self._head - self._size
            out = self._buffer[start : self._head].copy()
        else:
            tail = self._buffer[self._head - self._size :]
            head = self._buffer[: self._head]
            out = np.concatenate([tail, head], axis=0).copy()
        self._buffer.fill(0)
        self._size = 0
        self._head = 0
        return out

    def snapshot(self) -> np.ndarray:
        """Return a copy of the buffered samples without emptying them."""
        if self._size == 0:
            return np.zeros((0, self.channels), dtype=self.dtype)
        if self._head >= self._size:
            start = self._head - self._size
            return self._buffer[start : self._head].copy()
        tail = self._buffer[self._head - self._size :]
        head = self._buffer[: self._head]
        return np.concatenate([tail, head], axis=0).copy()

    def reset(self) -> None:
        self._buffer.fill(0)
        self._size = 0
        self._head = 0


__all__ = ["RingBuffer"]
