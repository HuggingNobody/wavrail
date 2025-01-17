"""Tests for the PCM16 codec."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from wavrail.audio._codec_pcm16 import decode, encode
from wavrail.errors import AudioError


def test_encode_decode_roundtrip() -> None:
    samples = np.array([0.0, 0.5, -0.5, 0.999, -0.999], dtype=np.float32)
    encoded = encode(samples)
    assert isinstance(encoded, bytes)
    assert len(encoded) == samples.size * 2
    decoded = decode(encoded, num_channels=1)
    assert decoded.dtype == np.float32
    np.testing.assert_allclose(decoded, samples, atol=1 / 32768.0)


def test_decode_stereo() -> None:
    raw = struct.pack("<4h", 0, 32767, -32768, 16384)
    decoded = decode(raw, num_channels=2)
    assert decoded.shape == (2, 2)
    assert decoded[0, 0] == pytest.approx(0.0)
    assert decoded[1, 0] == pytest.approx(-1.0, abs=1e-6)
    assert decoded[1, 1] == pytest.approx(0.5, abs=1e-6)


def test_encode_clips_overflow() -> None:
    samples = np.array([2.0, -2.0, 1.5, -1.5], dtype=np.float32)
    encoded = encode(samples)
    ints = struct.unpack(f"<{samples.size}h", encoded)
    assert max(ints) == 32767
    assert min(ints) == -32768


def test_decode_rejects_odd_length() -> None:
    with pytest.raises(AudioError):
        decode(b"\x00\x01\x02", num_channels=1)
