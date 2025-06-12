"""Cancellation semantics of the streaming pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.audio import synthesize_sine
from wavrail.errors import PipelineError
from wavrail.stream import (
    StreamingPipeline,
    deserialize_pipeline_state,
    serialize_pipeline_state,
)
from wavrail.vocab import Vocab


def _pipeline() -> StreamingPipeline:
    vocab = Vocab.from_labels(["<blank>", "a", "b"])
    return StreamingPipeline(sample_rate=16000, vocab=vocab, feature_dim=3)


def _samples(n: int) -> np.ndarray:
    wave = synthesize_sine(frequency_hz=440.0, duration_s=1.0, sample_rate=16000)
    return wave.astype(np.float32)[:n]


def test_feed_after_cancel_raises() -> None:
    pipeline = _pipeline()
    pipeline.feed_samples(_samples(800))
    pipeline.cancel()
    assert pipeline.is_cancelled
    with pytest.raises(PipelineError, match="cancelled"):
        pipeline.feed_samples(_samples(800))


def test_finalize_after_cancel_returns_partial_hypothesis() -> None:
    pipeline = _pipeline()
    emissions = pipeline.feed_samples(_samples(1600))
    pipeline.cancel()
    hyp = pipeline.finalize()
    # finalize may flush one pending candidate on top of what was
    # already emitted, but it never invents earlier tokens
    assert hyp.tokens[: len(emissions)] == tuple(emissions)
    assert len(hyp.tokens) <= len(emissions) + 1


def test_reset_clears_cancellation() -> None:
    pipeline = _pipeline()
    pipeline.cancel()
    pipeline.reset()
    assert not pipeline.is_cancelled
    pipeline.feed_samples(_samples(800))


def test_cancelled_flag_serializes() -> None:
    pipeline = _pipeline()
    pipeline.cancel()
    assert pipeline.state is not None
    payload = serialize_pipeline_state(pipeline.state)
    restored = deserialize_pipeline_state(payload)
    assert restored.cancelled is True


def test_cancel_before_any_feed() -> None:
    pipeline = _pipeline()
    pipeline.cancel()
    hyp = pipeline.finalize()
    assert hyp.tokens == ()
    assert hyp.text == ""


def test_double_cancel_is_idempotent() -> None:
    pipeline = _pipeline()
    pipeline.cancel()
    pipeline.cancel()
    assert pipeline.is_cancelled


def test_cancel_reset_cancel_cycle() -> None:
    pipeline = _pipeline()
    pipeline.cancel()
    pipeline.reset()
    pipeline.feed_samples(_samples(400))
    pipeline.cancel()
    with pytest.raises(PipelineError, match="cancelled"):
        pipeline.feed_samples(_samples(400))
