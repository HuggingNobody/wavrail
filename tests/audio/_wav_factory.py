"""Helpers for generating deterministic WAV byte streams in tests."""

from __future__ import annotations

import struct
from pathlib import Path


def write_pcm16_wav(
    path: Path,
    samples: list[int],
    sample_rate: int = 16000,
    num_channels: int = 1,
) -> None:
    """Write a single PCM16 mono or interleaved WAV file."""
    n = len(samples)
    data_bytes = struct.pack(f"<{n}h", *samples)
    fmt_chunk = struct.pack(
        "<HHIIHH",
        1,
        num_channels,
        sample_rate,
        sample_rate * num_channels * 2,
        num_channels * 2,
        16,
    )
    riff = struct.pack("<4sI4s", b"RIFF", 36 + len(data_bytes), b"WAVE")
    fmt_header = struct.pack("<4sI", b"fmt ", 16)
    data_header = struct.pack("<4sI", b"data", len(data_bytes))
    path.write_bytes(riff + fmt_header + fmt_chunk + data_header + data_bytes)


def write_float32_wav(path: Path, samples: list[float], sample_rate: int = 16000) -> None:
    """Write a single-channel Float32 WAV file."""
    data_bytes = struct.pack(f"<{len(samples)}f", *samples)
    fmt_chunk = struct.pack(
        "<HHIIHH",
        3,
        1,
        sample_rate,
        sample_rate * 4,
        4,
        32,
    )
    riff = struct.pack("<4sI4s", b"RIFF", 36 + len(data_bytes), b"WAVE")
    fmt_header = struct.pack("<4sI", b"fmt ", 16)
    data_header = struct.pack("<4sI", b"data", len(data_bytes))
    path.write_bytes(riff + fmt_header + fmt_chunk + data_header + data_bytes)
