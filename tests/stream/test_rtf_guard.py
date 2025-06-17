"""measure_rtf lifecycle guards."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.audio import synthesize_sine
from wavrail.errors import PipelineError
from wavrail.stream import StreamingPipeline, measure_rtf
from wavrail.vocab import Vocab


def _pipeline() -> StreamingPipeline:
    vocab = Vocab.from_labels(["<blank>", "a"])
    return StreamingPipeline(sample_rate=16000, vocab=vocab, feature_dim=2)


def _samples(n: int) -> np.ndarray:
    wave = synthesize_sine(frequency_hz=440.0, duration_s=1.0, sample_rate=16000)
    return wave.astype(np.float32)[:n]


def test_measure_rtf_rejects_cancelled_pipeline() -> None:
    pipeline = _pipeline()
    pipeline.cancel()
    with pytest.raises(PipelineError, match="cancelled"):
        measure_rtf(pipeline, _samples(1600))


def test_measure_rtf_result_keys() -> None:
    report = measure_rtf(_pipeline(), _samples(1600), chunk_size=400)
    assert set(report) == {"duration_seconds", "elapsed_seconds", "rtf"}
    assert report["duration_seconds"] == pytest.approx(0.1)
    assert report["elapsed_seconds"] >= 0.0
