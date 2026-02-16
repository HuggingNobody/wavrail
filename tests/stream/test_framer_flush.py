"""StreamingFramer.flush tail policies."""

from __future__ import annotations

import numpy as np

from wavrail.stream import StreamingFramer
from wavrail.types import FrameConfig

CFG = FrameConfig(frame_length=400, hop_length=160, center=False)
NO_OVERLAP = FrameConfig(frame_length=400, hop_length=400, center=False)


def test_drop_tail_matches_frame_signal():
    signal = np.arange(450, dtype=np.float32)
    framer = StreamingFramer(CFG)
    framer.push(signal)
    assert framer.flush().shape == (0, 400)
    assert framer.pending_samples == 0


def test_pad_tail_emits_zero_padded_frame():
    signal = np.arange(450, dtype=np.float32)
    framer = StreamingFramer(CFG)
    framer.push(signal)
    assert framer.pending_samples == 290  # hop-aligned overlap from sample 160
    tail = framer.flush(pad_tail=True)
    assert tail.shape == (1, 400)
    np.testing.assert_array_equal(tail[0, :290], signal[160:450])
    np.testing.assert_array_equal(tail[0, 290:], np.zeros(110, dtype=np.float32))


def test_pad_tail_on_empty_pending_is_empty():
    framer = StreamingFramer(NO_OVERLAP)
    framer.push(np.arange(400, dtype=np.float32))  # exactly one frame, no overlap
    assert framer.pending_samples == 0
    assert framer.flush(pad_tail=True).shape == (0, 400)


def test_flush_resets_counters_consistently():
    framer = StreamingFramer(CFG)
    framer.push(np.arange(500, dtype=np.float32))
    frames_before = framer.total_frames_emitted
    framer.flush(pad_tail=True)
    assert framer.pending_samples == 0
    assert framer.total_frames_emitted == frames_before + 1
    assert framer.total_input_samples == 500
