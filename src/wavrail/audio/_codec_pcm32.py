"""PCM32 codec: decode and encode signed 32-bit PCM audio.

PCM32 stores samples with 24 bits of dynamic range in the most
significant bytes of a 32-bit container. Decoded samples are
normalised to ``[-1, 1]`` using the ``1 / 2**31`` scale factor
used by ``soundfile``.
"""

from __future__ import annotations

import struct

import numpy as np

from wavrail.audio._wav_io import encode_pcm32, write_riff
from wavrail.errors import AudioError
from wavrail.types import AudioArray

_QUANT = 2147483648.0


def decode(data: bytes, num_channels: int) -> AudioArray:
    """Decode PCM32 bytes into a float32 array normalised to ``[-1, 1]``."""
    if num_channels <= 0:
        raise AudioError("num_channels must be positive")
    if len(data) % 4:
        raise AudioError("PCM32 byte length must be a multiple of 4")
    count = len(data) // 4
    samples = struct.unpack(f"<{count}i", data)
    arr = np.asarray(samples, dtype=np.int32).astype(np.float32) / _QUANT
    if num_channels > 1:
        arr = arr.reshape(-1, num_channels)
    return arr


def encode(samples: AudioArray) -> bytes:
    """Encode a float audio array into PCM32 bytes."""
    flat = np.asarray(samples, dtype=np.float32).reshape(-1)
    clipped = np.clip(flat, -1.0, 1.0 - 1 / _QUANT)
    ints = np.rint(clipped * _QUANT).astype(np.int64)
    return encode_pcm32(ints.tolist())


def write(
    path,
    samples: AudioArray,
    sample_rate: int,
    num_channels: int,
) -> None:
    """Encode ``samples`` and write a complete PCM32 WAV file."""
    write_riff(
        path,
        audio_format=0xFFFE,
        num_channels=num_channels,
        sample_rate=sample_rate,
        bits_per_sample=32,
        data_bytes=encode(samples),
    )


__all__ = ["decode", "encode", "write"]
