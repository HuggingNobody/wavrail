"""Tests for :func:`wavrail.audio.inspect_wav`."""

from __future__ import annotations

from pathlib import Path

import pytest

from wavrail.audio import inspect_wav
from wavrail.errors import FormatError

from ._wav_factory import write_float32_wav, write_pcm16_wav


def test_inspect_pcm16_mono(tmp_path: Path) -> None:
    p = tmp_path / "mono.wav"
    write_pcm16_wav(p, [0, 100, -100, 32767, -32768], sample_rate=8000)
    meta = inspect_wav(p)
    assert meta.sample_rate == 8000
    assert meta.num_channels == 1
    assert meta.sample_width == 2
    assert meta.num_samples == 5
    assert meta.duration_s == pytest.approx(5 / 8000)


def test_inspect_float32(tmp_path: Path) -> None:
    p = tmp_path / "f.wav"
    write_float32_wav(p, [0.0, 0.5, -0.5, 1.0], sample_rate=22050)
    meta = inspect_wav(p)
    assert meta.sample_rate == 22050
    assert meta.num_channels == 1
    assert meta.sample_width == 4
    assert meta.num_samples == 4


def test_inspect_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FormatError):
        inspect_wav(tmp_path / "absent.wav")


def test_inspect_invalid_riff(tmp_path: Path) -> None:
    p = tmp_path / "bad.wav"
    p.write_bytes(b"NOT A WAV FILE")
    with pytest.raises(FormatError):
        inspect_wav(p)
