"""Per-token emission latency tracking in the streaming pipeline."""

from __future__ import annotations

from wavrail.stream import (
    StreamingPipeline,
    deserialize_pipeline_state,
    serialize_pipeline_state,
)
from wavrail.vocab import Vocab

from ._helpers import sign_extractor, square_wave


def _pipeline(emission_delay: int) -> StreamingPipeline:
    return StreamingPipeline(
        sample_rate=16000,
        vocab=Vocab.from_labels(["<blank>", "a", "b"]),
        feature_dim=3,
        emission_delay=emission_delay,
        feature_extractor=sign_extractor,
    )


def test_latency_equals_confirmation_hops() -> None:
    pipeline = _pipeline(emission_delay=2)
    pipeline.feed_samples(square_wave())
    hyp = pipeline.finalize()
    assert hyp.tokens == (1, 2, 1)
    assert pipeline.state is not None
    assert pipeline.state.emission_latencies_frames == [1, 1, 1]


def test_zero_latency_with_immediate_emission() -> None:
    pipeline = _pipeline(emission_delay=1)
    pipeline.feed_samples(square_wave())
    hyp = pipeline.finalize()
    assert hyp.tokens == (1, 2, 1)
    assert pipeline.state is not None
    assert pipeline.state.emission_latencies_frames == [0, 0, 0]


def test_latencies_survive_serialization() -> None:
    pipeline = _pipeline(emission_delay=2)
    pipeline.feed_samples(square_wave())
    pipeline.finalize()
    assert pipeline.state is not None
    payload = serialize_pipeline_state(pipeline.state)
    restored = deserialize_pipeline_state(payload)
    assert restored.emission_latencies_frames == [1, 1, 1]


def test_latency_count_matches_token_count() -> None:
    pipeline = _pipeline(emission_delay=3)
    samples = square_wave()
    for start in range(0, samples.shape[0], 400):
        pipeline.feed_samples(samples[start : start + 400])
    hyp = pipeline.finalize()
    assert pipeline.state is not None
    assert len(pipeline.state.emission_latencies_frames) == len(hyp.tokens)
