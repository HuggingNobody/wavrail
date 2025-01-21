"""Float32 codec: decode and encode IEEE-754 WAV data.

Sample values are expected in the ``[-1, 1]`` range but are passed
through without scaling; the WAV spec leaves the valid range
implementation-defined and many tools exceed it.
"""

from __future__ import annotations

import struct

import numpy as np

from wavrail.audio._wav_io import encode_float32, write_riff
from wavrail.errors import AudioError
from wavrail.types import AudioArray


def decode(data: bytes, num_channels: int) -> AudioArray:
    """Decode Float32 bytes into a ``float32`` array."""
    if num_channels <= 0:
        raise AudioError("num_channels must be positive")
    if len(data) % 4:
        raise AudioError("Float32 byte length must be a multiple of 4")
    count = len(data) // 4
    samples = struct.unpack(f"<{count}f", data)
    arr = np.asarray(samples, dtype=np.float32)
    if num_channels > 1:
        arr = arr.reshape(-1, num_channels)
    return arr


def encode(samples: AudioArray) -> bytes:
    """Encode a float audio array as Float32 bytes.

    NaN and Inf values are clipped to ``[-1, 1]`` to keep the file
    well-formed.
    """
    flat = np.asarray(samples, dtype=np.float32).reshape(-1)
    cleaned = np.nan_to_num(flat, nan=0.0, posinf=1.0, neginf=-1.0)
    clipped = np.clip(cleaned, -1.0, 1.0)
    return encode_float32(clipped.tolist())


def write(
    path,
    samples: AudioArray,
    sample_rate: int,
    num_channels: int,
) -> None:
    """Encode ``samples`` and write a complete Float32 WAV file."""
    write_riff(
        path,
        audio_format=3,
        num_channels=num_channels,
        sample_rate=sample_rate,
        bits_per_sample=32,
        data_bytes=encode(samples),
    )


__all__ = ["decode", "encode", "write"]
