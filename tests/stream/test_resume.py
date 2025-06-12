"""Resume support: a paused pipeline continues where it stopped."""

from __future__ import annotations

import pytest

from wavrail.errors import PipelineError
from wavrail.stream import (
    StreamingPipeline,
    deserialize_pipeline_state,
    serialize_pipeline_state,
)
from wavrail.vocab import Vocab

from ._helpers import sign_extractor, square_wave


def _pipeline(sample_rate: int = 16000) -> StreamingPipeline:
    return StreamingPipeline(
        sample_rate=sample_rate,
        vocab=Vocab.from_labels(["<blank>", "a", "b"]),
        feature_dim=3,
        emission_delay=2,
        feature_extractor=sign_extractor,
    )


def test_resume_produces_same_tokens_as_uninterrupted() -> None:
    samples = square_wave()

    full = _pipeline()
    full.feed_samples(samples)
    expected = full.finalize().tokens
    assert expected == (1, 2, 1)

    first = _pipeline()
    first.feed_samples(samples[:1200])
    assert first.state is not None
    payload = serialize_pipeline_state(first.state)

    second = _pipeline()
    second.load_state(deserialize_pipeline_state(payload))
    second.feed_samples(samples[1200:])
    assert second.finalize().tokens == expected


def test_load_state_rejects_sample_rate_mismatch() -> None:
    source = _pipeline()
    assert source.state is not None
    payload = serialize_pipeline_state(source.state)
    other = _pipeline(sample_rate=8000)
    with pytest.raises(PipelineError, match="sample_rate"):
        other.load_state(deserialize_pipeline_state(payload))


def test_load_state_rejects_frame_geometry_mismatch() -> None:
    source = _pipeline()
    assert source.state is not None
    payload = serialize_pipeline_state(source.state)
    other = StreamingPipeline(
        sample_rate=16000,
        vocab=Vocab.from_labels(["<blank>", "a", "b"]),
        frame_length_ms=20.0,
        feature_dim=3,
    )
    with pytest.raises(PipelineError, match="geometry"):
        other.load_state(deserialize_pipeline_state(payload))


def test_load_state_rejects_vocab_mismatch() -> None:
    source = _pipeline()
    assert source.state is not None
    payload = serialize_pipeline_state(source.state)
    small = StreamingPipeline(
        sample_rate=16000,
        vocab=Vocab.from_labels(["<blank>", "a"]),
        feature_dim=2,
    )
    with pytest.raises(PipelineError, match="vocab"):
        small.load_state(deserialize_pipeline_state(payload))
