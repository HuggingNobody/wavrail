"""Tests for the PCM32 codec."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from wavrail.audio._codec_pcm32 import decode, encode
from wavrail.errors import AudioError


def test_encode_decode_roundtrip() -> None:
    samples = np.array([0.0, 0.25, -0.25, 0.999, -0.999], dtype=np.float32)
    encoded = encode(samples)
    assert isinstance(encoded, bytes)
    assert len(encoded) == samples.size * 4
    decoded = decode(encoded, num_channels=1)
    assert decoded.dtype == np.float32
    np.testing.assert_allclose(decoded, samples, atol=1 / 2147483648.0)


def test_decode_stereo_reshape() -> None:
    raw = struct.pack("<4i", 0, 1, -1, 2**30)
    decoded = decode(raw, num_channels=2)
    assert decoded.shape == (2, 2)
    assert decoded[0, 0] == pytest.approx(0.0)
    assert decoded[1, 1] == pytest.approx(0.5, abs=1e-5)


def test_encode_clips_overflow() -> None:
    samples = np.array([2.0, -2.0, 1.5, -1.5], dtype=np.float32)
    encoded = encode(samples)
    ints = struct.unpack(f"<{samples.size}i", encoded)
    assert max(ints) == 2147483647
    assert min(ints) == -2147483648


def test_decode_rejects_misaligned_bytes() -> None:
    with pytest.raises(AudioError):
        decode(b"\x00\x01\x02\x03\x04", num_channels=1)
