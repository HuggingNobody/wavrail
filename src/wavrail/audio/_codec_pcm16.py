"""PCM16 codec: decode and encode signed 16-bit PCM audio.

Decoded samples are normalised to the ``[-1, 1]`` float range with a
scale factor of ``1 / 32768`` (matching the ``np.int16`` -> float
convention used by :mod:`soundfile`). The encoder clips to the
``int16`` representable range before quantisation.
"""

from __future__ import annotations

import struct

import numpy as np

from wavrail.audio._wav_io import encode_pcm16, write_riff
from wavrail.errors import AudioError
from wavrail.types import AudioArray

_QUANT = 32768.0


def decode(data: bytes, num_channels: int) -> AudioArray:
    """Decode PCM16 bytes into a float32 array normalised to ``[-1, 1]``.

    Parameters
    ----------
    data:
        Raw little-endian PCM16 sample bytes.
    num_channels:
        Number of interleaved channels.
    """
    if num_channels <= 0:
        raise AudioError("num_channels must be positive")
    if len(data) % 2:
        raise AudioError("PCM16 byte length must be even")
    count = len(data) // 2
    samples = struct.unpack(f"<{count}h", data)
    arr = np.asarray(samples, dtype=np.int16).astype(np.float32) / _QUANT
    if num_channels > 1:
        arr = arr.reshape(-1, num_channels)
    return arr


def encode(samples: AudioArray) -> bytes:
    """Encode a float audio array into PCM16 bytes.

    Values outside ``[-1, 1]`` are clipped to the representable range.
    """
    flat = np.asarray(samples, dtype=np.float32).reshape(-1)
    clipped = np.clip(flat, -1.0, 1.0 - 1 / _QUANT)
    ints = np.rint(clipped * _QUANT).astype(np.int32)
    return encode_pcm16(ints.tolist())


def write(
    path,
    samples: AudioArray,
    sample_rate: int,
    num_channels: int,
) -> None:
    """Encode ``samples`` and write a complete WAV file."""
    write_riff(
        path,
        audio_format=1,
        num_channels=num_channels,
        sample_rate=sample_rate,
        bits_per_sample=16,
        data_bytes=encode(samples),
    )


__all__ = ["decode", "encode", "write"]
