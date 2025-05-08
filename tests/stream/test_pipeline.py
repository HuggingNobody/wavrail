"""Tests for the streaming ring buffer and pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.audio import synthesize_sine
from wavrail.stream import (
    RingBuffer,
    StreamingPipeline,
    measure_rtf,
)
from wavrail.vocab import Vocab


def test_ring_buffer_drain_returns_chronological() -> None:
    buf = RingBuffer(capacity=8, channels=1)
    buf.extend(np.arange(4, dtype=np.float32))
    buf.extend(np.arange(4, 8, dtype=np.float32))
    out = buf.drain()
    assert np.array_equal(out.reshape(-1), np.arange(8, dtype=np.float32))


def test_ring_buffer_wraps() -> None:
    buf = RingBuffer(capacity=4, channels=1)
    buf.extend(np.arange(4, dtype=np.float32))
    buf.drain()
    buf.extend(np.array([10, 11, 12, 13], dtype=np.float32))
    out = buf.drain()
    assert np.array_equal(out.reshape(-1), np.array([10, 11, 12, 13]))


def test_ring_buffer_rejects_oversize_push() -> None:
    buf = RingBuffer(capacity=4, channels=1)
    with pytest.raises(Exception, match="capacity"):
        buf.extend(np.arange(8, dtype=np.float32))


def test_pipeline_emits_during_feed() -> None:
    vocab = Vocab.from_labels(["<blank>", "a"])
    pipeline = StreamingPipeline(
        sample_rate=16000,
        vocab=vocab,
        emission_delay=2,
        feature_dim=2,
    )
    samples = synthesize_sine(frequency_hz=440.0, duration_s=0.5, sample_rate=16000).astype(
        np.float32
    )
    emissions: list[int] = []
    chunk = 800
    for start in range(0, samples.shape[0], chunk):
        emissions.extend(pipeline.feed_samples(samples[start : start + chunk]))
    pipeline.finalize()
    assert isinstance(emissions, list)


def test_measure_rtf_reports_positive() -> None:
    vocab = Vocab.from_labels(["<blank>", "a"])
    pipeline = StreamingPipeline(sample_rate=16000, vocab=vocab, feature_dim=2)
    samples = synthesize_sine(frequency_hz=440.0, duration_s=0.2, sample_rate=16000).astype(
        np.float32
    )
    rtf = measure_rtf(pipeline, samples, chunk_size=400)
    assert rtf["duration_seconds"] == pytest.approx(0.2, abs=1e-3)
