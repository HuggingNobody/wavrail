"""StreamingFramer push semantics and chunk-boundary stability."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.errors import PipelineError
from wavrail.features import frame_signal
from wavrail.stream import StreamingFramer
from wavrail.types import FrameConfig

CFG = FrameConfig(frame_length=64, hop_length=32, center=False)


def test_emits_frames_only_when_complete():
    framer = StreamingFramer(FrameConfig(frame_length=400, hop_length=160, center=False))
    out = framer.push(np.zeros(399, dtype=np.float32))
    assert out.shape == (0, 400)
    out = framer.push(np.ones(1, dtype=np.float32))
    assert out.shape == (1, 400)


def test_chunk_boundaries_do_not_shift_frames():
    framer = StreamingFramer(CFG)
    signal = np.arange(300, dtype=np.float32)
    chunks = [framer.push(signal[i : i + 37]) for i in range(0, 300, 37)]
    streamed = np.vstack(chunks)
    np.testing.assert_array_equal(streamed, frame_signal(signal, CFG))


def test_empty_push_is_noop():
    framer = StreamingFramer(CFG)
    assert framer.push(np.zeros(0, dtype=np.float32)).shape == (0, 64)
    assert framer.pending_samples == 0
    assert framer.total_input_samples == 0


def test_counters_track_progress():
    framer = StreamingFramer(CFG)
    framer.push(np.zeros(100, dtype=np.float32))  # frames: (100-64)//32+1 = 2
    assert framer.total_input_samples == 100
    assert framer.total_frames_emitted == 2
    assert framer.pending_samples == 100 - 2 * 32  # one hop consumed per frame


def test_rejects_center_config():
    with pytest.raises(PipelineError):
        StreamingFramer(FrameConfig(frame_length=64, hop_length=32))  # center defaults True
