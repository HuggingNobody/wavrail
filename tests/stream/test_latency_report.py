"""LatencyReport summarises frame-grid latency accounting."""

from __future__ import annotations

import pytest

from wavrail.stream import LatencyReport, StreamingPipeline
from wavrail.vocab import Vocab

from ._helpers import sign_extractor, square_wave


def _pipeline(emission_delay: int = 2) -> StreamingPipeline:
    return StreamingPipeline(
        sample_rate=16000,
        vocab=Vocab.from_labels(["<blank>", "a", "b"]),
        feature_dim=3,
        emission_delay=emission_delay,
        feature_extractor=sign_extractor,
    )


def test_report_values_match_geometry() -> None:
    pipeline = _pipeline(emission_delay=2)
    pipeline.feed_samples(square_wave())
    pipeline.finalize()
    report = pipeline.latency_report()
    assert isinstance(report, LatencyReport)
    assert report.tokens_emitted == 3
    assert report.algorithmic_ms == pytest.approx(35.0)
    # each token waited one confirmation hop of 160 samples = 10 ms
    assert report.mean_emission_ms == pytest.approx(10.0)
    assert report.max_emission_ms == pytest.approx(10.0)


def test_report_for_idle_pipeline_is_zero() -> None:
    report = _pipeline(emission_delay=1).latency_report()
    assert report.tokens_emitted == 0
    assert report.mean_emission_ms == 0.0
    assert report.max_emission_ms == 0.0
    assert report.algorithmic_ms == pytest.approx(25.0)
