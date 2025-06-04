"""Overflow policy and capacity accounting for the ring buffer."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.errors import PipelineError
from wavrail.stream import RingBuffer


def test_free_space_tracks_occupancy() -> None:
    buf = RingBuffer(capacity=8)
    assert buf.free_space == 8
    buf.extend(np.arange(6, dtype=np.float32))
    assert buf.free_space == 2
    buf.consume(4)
    assert buf.free_space == 6


def test_overwrite_policy_replaces_oldest() -> None:
    buf = RingBuffer(capacity=4, overflow="overwrite")
    buf.extend(np.arange(4, dtype=np.float32))
    buf.extend(np.array([9, 9], dtype=np.float32))
    assert buf.size == 4
    assert np.array_equal(buf.snapshot().reshape(-1), np.array([2, 3, 9, 9], dtype=np.float32))


def test_error_policy_raises_on_cumulative_overflow() -> None:
    buf = RingBuffer(capacity=4, overflow="error")
    buf.extend(np.arange(3, dtype=np.float32))
    with pytest.raises(PipelineError, match="overflow"):
        buf.extend(np.arange(2, dtype=np.float32))
    assert buf.size == 3


def test_single_push_over_capacity_always_raises() -> None:
    for policy in ("overwrite", "error"):
        buf = RingBuffer(capacity=4, overflow=policy)
        with pytest.raises(PipelineError, match="capacity"):
            buf.extend(np.arange(8, dtype=np.float32))


def test_rejects_unknown_overflow_policy() -> None:
    with pytest.raises(PipelineError, match="overflow"):
        RingBuffer(capacity=4, overflow="explode")
