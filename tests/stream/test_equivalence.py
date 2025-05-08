"""Streaming vs offline equivalence tests."""

from __future__ import annotations

import numpy as np

from wavrail.audio import synthesize_sine
from wavrail.ctc import greedy_decode
from wavrail.emit import decode_streaming
from wavrail.stream import StreamingPipeline
from wavrail.vocab import Vocab


def _vocab() -> Vocab:
    return Vocab.from_labels(["<blank>", "a", "b"])


def test_streaming_equals_offline_on_synthetic_posteriors() -> None:
    vocab = _vocab()
    posteriors = np.array(
        [
            [0.0, -5.0, -5.0],
            [0.0, -5.0, -5.0],
            [-5.0, 0.0, -5.0],
            [-5.0, 0.0, -5.0],
            [-5.0, -5.0, 0.0],
            [-5.0, -5.0, 0.0],
            [0.0, -5.0, -5.0],
        ],
        dtype=np.float64,
    )
    stream = decode_streaming(posteriors, vocab, emission_delay=2)
    offline = greedy_decode(posteriors, vocab)
    assert stream.tokens == offline.tokens


def test_pipeline_chunked_equals_full() -> None:
    vocab = _vocab()
    samples = synthesize_sine(frequency_hz=220.0, duration_s=0.3, sample_rate=16000).astype(
        np.float32
    )
    pipeline = StreamingPipeline(sample_rate=16000, vocab=vocab, emission_delay=2, feature_dim=3)
    chunk = 400
    for start in range(0, samples.shape[0], chunk):
        pipeline.feed_samples(samples[start : start + chunk])
    chunked = pipeline.finalize()
    pipeline2 = StreamingPipeline(sample_rate=16000, vocab=vocab, emission_delay=2, feature_dim=3)
    pipeline2.feed_samples(samples)
    full = pipeline2.finalize()
    assert chunked.tokens == full.tokens
