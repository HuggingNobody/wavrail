"""Exact-size consumption from the ring buffer."""

from __future__ import annotations

import numpy as np

from wavrail.stream import RingBuffer


def test_consume_drops_oldest_samples() -> None:
    buf = RingBuffer(capacity=8)
    buf.extend(np.arange(6, dtype=np.float32))
    assert buf.consume(2) == 2
    assert buf.size == 4
    assert np.array_equal(buf.snapshot().reshape(-1), np.arange(2, 6, dtype=np.float32))


def test_consume_more_than_size_empties_buffer() -> None:
    buf = RingBuffer(capacity=8)
    buf.extend(np.arange(3, dtype=np.float32))
    assert buf.consume(10) == 3
    assert buf.size == 0


def test_consume_zero_and_negative_are_noops() -> None:
    buf = RingBuffer(capacity=8)
    buf.extend(np.arange(4, dtype=np.float32))
    assert buf.consume(0) == 0
    assert buf.consume(-5) == 0
    assert buf.size == 4


def test_consume_across_wrap_keeps_chronology() -> None:
    buf = RingBuffer(capacity=4)
    buf.extend(np.arange(4, dtype=np.float32))
    buf.consume(3)
    buf.extend(np.arange(4, 7, dtype=np.float32))
    assert np.array_equal(buf.snapshot().reshape(-1), np.array([3, 4, 5, 6], dtype=np.float32))
