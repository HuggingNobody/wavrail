"""The serialisable state must capture the pending audio tail."""

from __future__ import annotations

import numpy as np

from wavrail.audio import synthesize_sine
from wavrail.stream import StreamingPipeline, serialize_pipeline_state
from wavrail.vocab import Vocab


def _pipeline() -> StreamingPipeline:
    vocab = Vocab.from_labels(["<blank>", "a"])
    return StreamingPipeline(sample_rate=16000, vocab=vocab, feature_dim=2)


def _samples(n: int) -> np.ndarray:
    wave = synthesize_sine(frequency_hz=440.0, duration_s=1.0, sample_rate=16000)
    return wave.astype(np.float32)[:n]


def test_state_captures_pending_tail() -> None:
    pipeline = _pipeline()
    pipeline.feed_samples(_samples(500))
    assert pipeline.state is not None
    payload = serialize_pipeline_state(pipeline.state)
    assert len(payload["buffer_samples"]) == 340
    assert pipeline.buffered_samples == 340


def test_state_captures_tail_below_frame_length() -> None:
    pipeline = _pipeline()
    pipeline.feed_samples(_samples(200))
    assert pipeline.state is not None
    payload = serialize_pipeline_state(pipeline.state)
    assert len(payload["buffer_samples"]) == 200


def test_state_empty_after_finalize() -> None:
    pipeline = _pipeline()
    pipeline.feed_samples(_samples(800))
    pipeline.finalize()
    assert pipeline.state is not None
    payload = serialize_pipeline_state(pipeline.state)
    assert payload["buffer_samples"] == []
