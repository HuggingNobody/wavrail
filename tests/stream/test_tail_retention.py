"""The pipeline must retain the unframed tail between feeds."""

from __future__ import annotations

import numpy as np

from wavrail.audio import synthesize_sine
from wavrail.stream import StreamingPipeline
from wavrail.vocab import Vocab


def _pipeline() -> StreamingPipeline:
    vocab = Vocab.from_labels(["<blank>", "a"])
    return StreamingPipeline(sample_rate=16000, vocab=vocab, feature_dim=2)


def _samples(n: int) -> np.ndarray:
    wave = synthesize_sine(frequency_hz=440.0, duration_s=1.0, sample_rate=16000)
    return wave.astype(np.float32)[:n]


def test_no_frames_until_frame_length_reached() -> None:
    pipeline = _pipeline()
    pipeline.feed_samples(_samples(200))
    assert pipeline.buffered_samples == 200
    pipeline.feed_samples(_samples(200))
    # 400 samples form one frame; consuming it advances by one hop (160).
    assert pipeline.buffered_samples == 240


def test_tail_below_frame_length_is_retained() -> None:
    pipeline = _pipeline()
    pipeline.feed_samples(_samples(500))
    assert pipeline.buffered_samples == 340


def test_full_session_leaves_exact_remainder() -> None:
    pipeline = _pipeline()
    samples = _samples(4800)
    for start in range(0, samples.shape[0], 400):
        pipeline.feed_samples(samples[start : start + 400])
    # 28 frames * 160-sample hop = 4480 consumed, 320 remain buffered.
    assert pipeline.buffered_samples == 320
