"""Framer state serialization, resumption, and golden key contract."""

from __future__ import annotations

import json

import numpy as np
import pytest

from wavrail.errors import PipelineError
from wavrail.stream import (
    FRAMER_STATE_VERSION,
    StreamingFramer,
    deserialize_framer_state,
    serialize_framer_state,
)
from wavrail.types import FrameConfig

CFG = FrameConfig(frame_length=64, hop_length=32, center=False)


def _framed_signal() -> np.ndarray:
    return np.random.default_rng(21).standard_normal(500).astype(np.float32)


def test_roundtrip_resumes_mid_signal():
    signal = _framed_signal()
    framer = StreamingFramer(CFG)
    first = framer.push(signal[:250])
    resumed = deserialize_framer_state(serialize_framer_state(framer))

    np.testing.assert_array_equal(resumed.pending_array(), framer.pending_array())
    before = resumed.total_frames_emitted

    a = np.vstack([first, framer.push(signal[250:])])
    b = np.vstack([first, resumed.push(signal[250:])])
    np.testing.assert_array_equal(a, b)
    assert resumed.total_input_samples == framer.total_input_samples
    assert resumed.total_frames_emitted == before + (b.shape[0] - first.shape[0])


def test_survives_json_roundtrip():
    framer = StreamingFramer(CFG)
    framer.push(_framed_signal()[:100])
    payload = json.loads(json.dumps(serialize_framer_state(framer)))
    resumed = deserialize_framer_state(payload)
    np.testing.assert_array_equal(resumed.pending_array(), framer.pending_array())


def test_golden_key_contract():
    payload = serialize_framer_state(StreamingFramer(CFG))
    assert payload["schema_version"] == FRAMER_STATE_VERSION == 1
    assert sorted(payload) == [
        "center",
        "frame_length",
        "hop_length",
        "pad_mode",
        "pending",
        "schema_version",
        "total_frames_emitted",
        "total_input_samples",
    ]


def test_rejects_unknown_schema_version():
    payload = serialize_framer_state(StreamingFramer(CFG))
    payload["schema_version"] = 99
    with pytest.raises(PipelineError):
        deserialize_framer_state(payload)


def test_restore_rejects_inconsistent_state():
    framer = StreamingFramer(CFG)
    with pytest.raises(PipelineError):
        framer.restore(np.zeros(64, dtype=np.float32), 64, 0)  # pending >= frame_length
    with pytest.raises(PipelineError):
        framer.restore(np.zeros(10, dtype=np.float32), 5, 0)  # counters below pending
