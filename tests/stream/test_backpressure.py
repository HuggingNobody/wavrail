"""Backpressure: buffer overflow surfaces to the producer."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.errors import PipelineError
from wavrail.stream import StreamingPipeline
from wavrail.vocab import Vocab


def _pipeline(**kwargs) -> StreamingPipeline:
    vocab = Vocab.from_labels(["<blank>", "a"])
    return StreamingPipeline(sample_rate=1000, vocab=vocab, feature_dim=2, **kwargs)


def test_error_policy_raises_when_chunk_exceeds_free_space() -> None:
    pipeline = _pipeline(buffer_capacity_seconds=0.05)  # capacity: 100 samples
    pipeline.feed_samples(np.zeros(95, dtype=np.float32))
    assert pipeline.buffered_samples == 15
    with pytest.raises(PipelineError, match="overflow"):
        pipeline.feed_samples(np.zeros(90, dtype=np.float32))


def test_error_policy_leaves_buffer_intact_after_raise() -> None:
    pipeline = _pipeline(buffer_capacity_seconds=0.05)
    pipeline.feed_samples(np.zeros(95, dtype=np.float32))
    with pytest.raises(PipelineError, match="overflow"):
        pipeline.feed_samples(np.zeros(90, dtype=np.float32))
    assert pipeline.buffered_samples == 15


def test_overwrite_policy_keeps_accepting_chunks() -> None:
    pipeline = _pipeline(buffer_capacity_seconds=0.05, overflow="overwrite")
    for _ in range(4):
        pipeline.feed_samples(np.zeros(95, dtype=np.float32))
    assert pipeline.buffered_samples <= 100
