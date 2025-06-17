"""Algorithmic latency accounting derived from frame geometry."""

from __future__ import annotations

import pytest

from wavrail.stream import StreamingPipeline
from wavrail.vocab import Vocab


def _pipeline(emission_delay: int = 1) -> StreamingPipeline:
    vocab = Vocab.from_labels(["<blank>", "a", "b"])
    return StreamingPipeline(
        sample_rate=16000,
        vocab=vocab,
        feature_dim=3,
        emission_delay=emission_delay,
    )


def test_latency_without_confirmation_hop() -> None:
    # one 400-sample frame at 16 kHz = 25 ms
    assert _pipeline(emission_delay=1).algorithmic_latency_ms == pytest.approx(25.0)


def test_latency_matches_frame_geometry() -> None:
    # 400 samples + 1 confirmation hop of 160 = 560 samples -> 35 ms
    assert _pipeline(emission_delay=2).algorithmic_latency_ms == pytest.approx(35.0)


def test_latency_grows_monotonically_with_delay() -> None:
    latencies = [_pipeline(emission_delay=d).algorithmic_latency_ms for d in (1, 2, 3, 4)]
    assert latencies == sorted(latencies)
    assert len(set(latencies)) == 4
