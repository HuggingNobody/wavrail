"""StreamingFramer lookahead and latency accounting."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.errors import PipelineError
from wavrail.stream import StreamingFramer
from wavrail.types import FrameConfig

CFG = FrameConfig(frame_length=400, hop_length=160, center=False)


def test_lookahead_is_the_frame_overlap():
    assert StreamingFramer(CFG).lookahead_samples == 240


def test_emission_latency_is_one_frame():
    framer = StreamingFramer(CFG)
    assert framer.emission_latency_ms(16000) == pytest.approx(25.0)


def test_max_buffered_bound_holds_under_random_pushes():
    framer = StreamingFramer(CFG)
    rng = np.random.default_rng(17)
    assert framer.max_buffered_samples == 399
    for _ in range(30):
        framer.push(rng.standard_normal(int(rng.integers(1, 900))).astype(np.float32))
        assert framer.pending_samples <= framer.max_buffered_samples


def test_latency_validation():
    with pytest.raises(PipelineError):
        StreamingFramer(CFG).emission_latency_ms(0)
