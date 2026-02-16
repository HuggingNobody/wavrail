"""Seeded property: streamed frames equal offline frame_signal exactly.

Also pins the bounded-buffer invariant: after every push the pending
buffer holds fewer than ``frame_length`` samples.
"""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.features import frame_signal
from wavrail.stream import StreamingFramer
from wavrail.types import FrameConfig


@pytest.mark.parametrize("seed", range(5))
def test_streaming_equals_offline(seed: int) -> None:
    rng = np.random.default_rng(seed)
    num_samples = int(rng.integers(500, 4000))
    frame_length = int(rng.integers(64, 256))
    hop_length = int(rng.integers(32, frame_length + 1))
    chunk = int(rng.integers(1, 300))
    cfg = FrameConfig(frame_length=frame_length, hop_length=hop_length, center=False)

    signal = rng.standard_normal(num_samples).astype(np.float32)
    framer = StreamingFramer(cfg)
    emitted = []
    for start in range(0, num_samples, chunk):
        frames = framer.push(signal[start : start + chunk])
        emitted.append(frames)
        assert framer.pending_samples < frame_length  # bounded buffer invariant
    streamed = np.vstack(emitted)
    np.testing.assert_array_equal(streamed, frame_signal(signal, cfg))


def test_single_sample_pushes_still_match():
    cfg = FrameConfig(frame_length=32, hop_length=16, center=False)
    signal = np.arange(200, dtype=np.float32)
    framer = StreamingFramer(cfg)
    emitted = np.vstack([framer.push(signal[i : i + 1]) for i in range(200)])
    np.testing.assert_array_equal(emitted, frame_signal(signal, cfg))
