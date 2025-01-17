"""Low-level RIFF / WAV byte construction.

Used by the per-encoding codec modules to assemble the header and
data chunks. The helpers are deliberately small and return raw
bytes; the higher level :mod:`wavrail.audio.wav` module composes
them.
"""

from __future__ import annotations

import struct
from pathlib import Path

from wavrail.errors import AudioError


def write_riff(
    path: Path,
    audio_format: int,
    num_channels: int,
    sample_rate: int,
    bits_per_sample: int,
    data_bytes: bytes,
) -> None:
    """Write a complete RIFF/WAVE file using the supplied chunks."""
    p = Path(path)
    if p.suffix.lower() != ".wav":
        raise AudioError("write_riff requires a .wav path")
    if sample_rate <= 0:
        raise AudioError("sample_rate must be positive")
    if num_channels <= 0:
        raise AudioError("num_channels must be positive")
    if bits_per_sample not in {16, 24, 32}:
        raise AudioError(f"unsupported bits_per_sample: {bits_per_sample}")
    bytes_per_sample = bits_per_sample // 8
    fmt_chunk = struct.pack(
        "<HHIIHH",
        audio_format,
        num_channels,
        sample_rate,
        sample_rate * num_channels * bytes_per_sample,
        num_channels * bytes_per_sample,
        bits_per_sample,
    )
    riff = struct.pack("<4sI4s", b"RIFF", 36 + len(data_bytes), b"WAVE")
    fmt_header = struct.pack("<4sI", b"fmt ", 16)
    data_header = struct.pack("<4sI", b"data", len(data_bytes))
    p.write_bytes(riff + fmt_header + fmt_chunk + data_header + data_bytes)


def encode_pcm16(samples: list[int]) -> bytes:
    """Pack a list of signed 16-bit samples into little-endian bytes."""
    return struct.pack(f"<{len(samples)}h", *samples)


def encode_pcm32(samples: list[int]) -> bytes:
    """Pack a list of signed 32-bit samples into little-endian bytes."""
    return struct.pack(f"<{len(samples)}i", *samples)


def encode_float32(samples: list[float]) -> bytes:
    """Pack a list of IEEE-754 floats into little-endian bytes."""
    return struct.pack(f"<{len(samples)}f", *samples)


__all__ = ["encode_float32", "encode_pcm16", "encode_pcm32", "write_riff"]
