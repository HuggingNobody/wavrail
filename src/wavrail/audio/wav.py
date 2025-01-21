"""Public WAV read/write entry points.

The dispatcher selects between the :mod:`wavrail.audio._codec_pcm16`,
:mod:`wavrail.audio._codec_pcm32`, and
:mod:`wavrail.audio._codec_float32` codec modules based on the
WAV header's ``audio_format`` field. The encoder takes an explicit
``encoding`` argument (``pcm16``, ``pcm32``, or ``float32``).

Only the three supported encodings are accepted. Anything else
raises :class:`wavrail.errors.FormatError` so the failure surfaces
at the I/O boundary, not inside a downstream consumer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from wavrail.audio import _codec_float32, _codec_pcm16, _codec_pcm32
from wavrail.audio.info import _read_header
from wavrail.errors import AudioError, FormatError
from wavrail.types import AudioArray, AudioMeta

Encoding = Literal["pcm16", "pcm32", "float32"]


def _select_decoder(audio_format: int, bits_per_sample: int):
    if audio_format == 1:
        if bits_per_sample != 16:
            raise FormatError(f"PCM format with {bits_per_sample}-bit samples is not supported")
        return _codec_pcm16.decode
    if audio_format == 3:
        if bits_per_sample != 32:
            raise FormatError(f"Float format with {bits_per_sample}-bit samples is not supported")
        return _codec_float32.decode
    if audio_format == 0xFFFE:
        if bits_per_sample == 16:
            return _codec_pcm16.decode
        if bits_per_sample == 32:
            return _codec_pcm32.decode
    raise FormatError(
        f"Unsupported WAV encoding audio_format={audio_format}, bits={bits_per_sample}",
    )


def read_wav(path: str | Path) -> tuple[AudioArray, AudioMeta]:
    """Decode a WAV file into a ``float32`` array plus metadata."""
    p = Path(path)
    if not p.exists():
        raise AudioError(f"{path} does not exist")
    audio_format, num_channels, sample_rate, bits_per_sample, data_size = _read_header(p)
    decoder = _select_decoder(audio_format, bits_per_sample)
    with p.open("rb") as handle:
        handle.seek(44, 0)
        data = handle.read(data_size)
    samples = decoder(data, num_channels)
    sample_width = bits_per_sample // 8
    num_samples = samples.shape[0]
    duration_s = num_samples / sample_rate
    meta = AudioMeta(
        sample_rate=sample_rate,
        num_channels=num_channels,
        sample_width=sample_width,
        duration_s=duration_s,
        num_samples=num_samples,
    )
    return samples, meta


def write_wav(
    path: str | Path,
    samples: AudioArray,
    sample_rate: int,
    encoding: Encoding = "pcm16",
) -> None:
    """Encode ``samples`` and write a WAV file with the requested encoding."""
    p = Path(path)
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 1:
        num_channels = 1
    elif arr.ndim == 2:
        num_channels = arr.shape[1]
    else:
        raise AudioError("samples must be 1-D or 2-D")
    if sample_rate <= 0:
        raise AudioError("sample_rate must be positive")
    if encoding == "pcm16":
        _codec_pcm16.write(p, arr, sample_rate, num_channels)
    elif encoding == "pcm32":
        _codec_pcm32.write(p, arr, sample_rate, num_channels)
    elif encoding == "float32":
        _codec_float32.write(p, arr, sample_rate, num_channels)
    else:
        raise AudioError(f"Unknown encoding: {encoding}")


__all__ = ["read_wav", "write_wav"]
