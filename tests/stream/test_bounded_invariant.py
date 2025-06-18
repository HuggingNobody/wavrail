"""The pipeline buffer must stay bounded over long feeding sessions."""

from __future__ import annotations

import numpy as np

from wavrail.stream import StreamingPipeline
from wavrail.vocab import Vocab


def _pipeline() -> StreamingPipeline:
    vocab = Vocab.from_labels(["<blank>", "a"])
    return StreamingPipeline(
        sample_rate=1000,
        vocab=vocab,
        feature_dim=2,
        buffer_capacity_seconds=0.05,  # capacity: 100 samples
    )


def test_buffer_never_exceeds_capacity() -> None:
    pipeline = _pipeline()
    rng = np.random.default_rng(11)
    for _ in range(300):
        n = int(rng.integers(1, 60))
        pipeline.feed_samples(rng.standard_normal(n).astype(np.float32))
        assert pipeline.buffered_samples <= pipeline.buffer.capacity


def test_error_policy_survives_steady_feeding() -> None:
    # consumption keeps pace, so the default "error" overflow policy
    # must never trigger for chunks below capacity
    pipeline = _pipeline()
    rng = np.random.default_rng(12)
    for _ in range(300):
        n = int(rng.integers(1, 60))
        pipeline.feed_samples(rng.standard_normal(n).astype(np.float32))
    assert pipeline.buffered_samples < 100


def test_ingested_sample_accounting() -> None:
    pipeline = _pipeline()
    rng = np.random.default_rng(13)
    total = 0
    for _ in range(120):
        n = int(rng.integers(1, 60))
        pipeline.feed_samples(np.zeros(n, dtype=np.float32))
        total += n
    assert pipeline.state is not None
    assert pipeline.state.total_samples_consumed == total
