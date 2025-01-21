"""Tests for the public :mod:`wavrail.audio.wav` entry points."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from wavrail.audio import read_wav, write_wav
from wavrail.errors import AudioError, FormatError


def _roundtrip(encoding: str, sample_rate: int, samples: np.ndarray) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"clip.{encoding}.wav"
        write_wav(path, samples, sample_rate, encoding=encoding)
        decoded, meta = read_wav(path)
    assert meta.sample_rate == sample_rate
    assert meta.num_channels == 1
    if encoding == "pcm16":
        np.testing.assert_allclose(decoded, samples, atol=1 / 32768.0)
    elif encoding == "pcm32":
        np.testing.assert_allclose(decoded, samples, atol=1 / 2147483648.0)
    else:
        np.testing.assert_array_equal(decoded, samples)


def test_roundtrip_pcm16() -> None:
    samples = np.array([0.0, 0.5, -0.5, 0.9, -0.9], dtype=np.float32)
    _roundtrip("pcm16", 16000, samples)


def test_roundtrip_pcm32() -> None:
    samples = np.array([0.0, 0.25, -0.25, 0.999, -0.999], dtype=np.float32)
    _roundtrip("pcm32", 22050, samples)


def test_roundtrip_float32() -> None:
    samples = np.array([0.0, 0.5, -0.5, 0.9, -0.9], dtype=np.float32)
    _roundtrip("float32", 44100, samples)


def test_read_wav_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AudioError):
        read_wav(tmp_path / "absent.wav")


def test_write_wav_rejects_unknown_encoding(tmp_path: Path) -> None:
    samples = np.zeros(8, dtype=np.float32)
    with pytest.raises(AudioError):
        write_wav(tmp_path / "x.wav", samples, 16000, encoding="mu-law")  # type: ignore[arg-type]


def test_read_wav_rejects_unsupported_encoding(tmp_path: Path) -> None:
    p = tmp_path / "alaw.wav"
    fmt = b"fmt " + (16).to_bytes(4, "little") + b"\x06\x00\x01\x00"
    data = b"data" + b"\x00\x00\x00\x00" + b"\x00" * 32
    p.write_bytes(b"RIFF" + (36 + len(data)).to_bytes(4, "little") + b"WAVE" + fmt + data)
    with pytest.raises(FormatError):
        read_wav(p)


def test_write_wav_stereo(tmp_path: Path) -> None:
    samples = np.array([[0.1, 0.2], [0.3, 0.4], [-0.1, -0.2]], dtype=np.float32)
    p = tmp_path / "stereo.wav"
    write_wav(p, samples, 16000)
    decoded, meta = read_wav(p)
    assert meta.num_channels == 2
    np.testing.assert_allclose(decoded, samples, atol=1 / 32768.0)
