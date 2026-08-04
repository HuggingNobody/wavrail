"""Bounded ring buffer for streaming audio frames.

The buffer is backed by a NumPy array of fixed capacity. Old
samples are overwritten when the buffer wraps around. The
buffer never grows unboundedly. Cumulative overflow follows the
``overflow`` policy: ``"overwrite"`` silently replaces the
oldest samples, while ``"error"`` raises so the producer can
apply backpressure. A single push larger than the whole
capacity always raises.
"""

from __future__ import annotations

import numpy as np

from wavrail.errors import PipelineError as _BufferError

_DEFAULT_DTYPE = np.dtype(np.float32)


class RingBuffer:
    """Fixed-capacity NumPy-backed ring buffer."""

    def __init__(
        self,
        capacity: int,
        channels: int = 1,
        dtype: np.dtype = _DEFAULT_DTYPE,
        overflow: str = "overwrite",
    ) -> None:
        if capacity < 1:
            raise _BufferError("capacity must be >= 1")
        if channels < 1:
            raise _BufferError("channels must be >= 1")
        if overflow not in {"overwrite", "error"}:
            raise _BufferError("overflow must be 'overwrite' or 'error'")
        self.overflow = overflow
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
    def free_space(self) -> int:
        """Samples that fit before the overflow policy kicks in."""
        return self.capacity - self._size

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
        if self.overflow == "error" and self._size + n > self.capacity:
            raise _BufferError(
                f"buffer overflow: {n} samples do not fit in {self.free_space} free slots"
            )
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

    def consume(self, n: int) -> int:
        """Discard the oldest ``n`` samples and return how many were dropped.

        Asking for more samples than are buffered empties the buffer
        without raising; the return value reports the number actually
        dropped so callers can detect an underrun.
        """
        if n <= 0:
            return 0
        dropped = min(n, self._size)
        self._size -= dropped
        return dropped

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
