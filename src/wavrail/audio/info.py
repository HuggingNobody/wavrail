"""Lightweight WAV header reader.

Only the header is parsed; sample data is not decoded. The returned
:class:`wavrail.types.AudioMeta` reports the sample rate, channel
count, bit depth, sample count, and duration.
"""

from __future__ import annotations

import struct
from pathlib import Path

from wavrail.errors import FormatError
from wavrail.types import AudioMeta


def _read_header(path: Path) -> tuple[int, int, int, int, int]:
    with path.open("rb") as handle:
        riff = handle.read(12)
        if len(riff) < 12 or riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise FormatError(f"{path} is not a RIFF/WAVE file")
        sample_rate = 0
        num_channels = 0
        bits_per_sample = 0
        audio_format = 0
        while True:
            header = handle.read(8)
            if len(header) < 8:
                raise FormatError(f"{path} truncated before data chunk")
            tag = header[:4]
            size = struct.unpack("<I", header[4:8])[0]
            if tag == b"fmt ":
                fmt = handle.read(size)
                if len(fmt) < 16:
                    raise FormatError(f"{path} has malformed fmt chunk")
                audio_format, num_channels = struct.unpack("<HH", fmt[:4])
                sample_rate = struct.unpack("<I", fmt[4:8])[0]
                bits_per_sample = struct.unpack("<H", fmt[14:16])[0]
                if size % 2:
                    handle.read(1)
            elif tag == b"data":
                return audio_format, num_channels, sample_rate, bits_per_sample, size
            else:
                if size % 2:
                    size += 1
                handle.read(size)


def inspect_wav(path: str | Path) -> AudioMeta:
    """Return audio metadata without decoding sample data.

    Parameters
    ----------
    path:
        Filesystem path to the WAV file.

    Returns
    -------
    AudioMeta
        Sample rate, channel count, sample width, sample count, duration.
    """
    p = Path(path)
    if not p.exists():
        raise FormatError(f"{path} does not exist")
    audio_format, num_channels, sample_rate, bits_per_sample, data_size = _read_header(p)
    _ = audio_format
    if sample_rate <= 0 or num_channels <= 0 or bits_per_sample <= 0:
        raise FormatError(f"{path} has invalid fmt fields")
    sample_width = bits_per_sample // 8
    num_samples = data_size // (sample_width * num_channels)
    duration_s = num_samples / sample_rate
    return AudioMeta(
        sample_rate=sample_rate,
        num_channels=num_channels,
        sample_width=sample_width,
        duration_s=duration_s,
        num_samples=num_samples,
    )


__all__ = ["inspect_wav"]
