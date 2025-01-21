"""Tests for the Float32 codec."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from wavrail.audio._codec_float32 import decode, encode
from wavrail.errors import AudioError


def test_encode_decode_roundtrip() -> None:
    samples = np.array([0.0, 0.5, -0.5, 0.9, -0.1], dtype=np.float32)
    encoded = encode(samples)
    assert len(encoded) == samples.size * 4
    decoded = decode(encoded, num_channels=1)
    np.testing.assert_array_equal(decoded, samples)


def test_decode_stereo_reshape() -> None:
    raw = struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)
    decoded = decode(raw, num_channels=2)
    assert decoded.shape == (2, 2)
    np.testing.assert_array_equal(decoded, np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))


def test_encode_cleans_nan() -> None:
    samples = np.array([0.0, np.nan, np.inf, -np.inf, 0.25], dtype=np.float32)
    encoded = encode(samples)
    ints = struct.unpack(f"<{samples.size}f", encoded)
    assert ints[0] == 0.0
    assert ints[1] == 0.0
    assert ints[2] == 1.0
    assert ints[3] == -1.0
    assert ints[4] == pytest.approx(0.25, abs=1e-6)


def test_decode_rejects_misaligned_bytes() -> None:
    with pytest.raises(AudioError):
        decode(b"\x00\x01\x02", num_channels=1)
