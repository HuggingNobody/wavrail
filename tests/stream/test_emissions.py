"""Deterministic end-to-end emissions through the streaming pipeline."""

from __future__ import annotations

import pytest

from wavrail.stream import StreamingPipeline
from wavrail.vocab import Vocab

from ._helpers import sign_extractor, square_wave


def _pipeline(emission_delay: int = 1) -> StreamingPipeline:
    return StreamingPipeline(
        sample_rate=16000,
        vocab=Vocab.from_labels(["<blank>", "a", "b"]),
        feature_dim=3,
        emission_delay=emission_delay,
        feature_extractor=sign_extractor,
    )


def test_square_wave_emits_expected_pattern() -> None:
    pipeline = _pipeline()
    pipeline.feed_samples(square_wave())
    hyp = pipeline.finalize()
    assert hyp.tokens == (1, 2, 1)
    assert hyp.text == "aba"


def test_emission_delay_preserves_token_pattern() -> None:
    pipeline = _pipeline(emission_delay=2)
    samples = square_wave()
    for start in range(0, samples.shape[0], 400):
        pipeline.feed_samples(samples[start : start + 400])
    hyp = pipeline.finalize()
    assert hyp.tokens == (1, 2, 1)


@pytest.mark.parametrize("chunk", [160, 400, 401, 997, 2400])
def test_chunking_does_not_change_tokens(chunk: int) -> None:
    samples = square_wave()

    def run(size: int) -> tuple[int, ...]:
        pipeline = _pipeline()
        for start in range(0, samples.shape[0], size):
            pipeline.feed_samples(samples[start : start + size])
        return pipeline.finalize().tokens

    assert run(chunk) == (1, 2, 1)
