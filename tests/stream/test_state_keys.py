"""Golden key sets for serialised stream state; drift must be deliberate."""

from __future__ import annotations

import numpy as np

from wavrail.stream import StreamingPipeline, serialize_pipeline_state
from wavrail.vocab import Vocab


def _payload() -> dict:
    vocab = Vocab.from_labels(["<blank>", "a"])
    pipeline = StreamingPipeline(sample_rate=16000, vocab=vocab, feature_dim=2)
    pipeline.feed_samples(np.zeros(500, dtype=np.float32))
    assert pipeline.state is not None
    return serialize_pipeline_state(pipeline.state)


def test_pipeline_state_keys_are_stable() -> None:
    assert set(_payload()) == {
        "buffer_samples",
        "total_samples_consumed",
        "decoder_state",
        "feature_lookahead_samples",
        "sample_rate",
        "frame_length_samples",
        "hop_length_samples",
        "cancelled",
        "emission_latencies_frames",
    }


def test_decoder_state_keys_are_stable() -> None:
    assert set(_payload()["decoder_state"]) == {
        "vocab_size",
        "blank_index",
        "emission_delay",
        "max_symbols_per_step",
        "tokens",
        "frame_indices",
        "log_probabilities",
        "candidate_symbol",
        "candidate_count",
        "candidate_frame",
        "frames_consumed",
    }
