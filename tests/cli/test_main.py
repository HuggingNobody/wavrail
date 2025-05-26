"""Tests for the CLI dispatcher."""

from __future__ import annotations

import json

import pytest

from wavrail.audio import write_wav
from wavrail.audio.synthetic import synthesize_sine
from wavrail.cli import EXIT_OK, EXIT_RUNTIME_ERROR, main


@pytest.fixture
def wav_path(tmp_path) -> str:
    samples = synthesize_sine(frequency_hz=440.0, duration_s=0.2, sample_rate=16000)
    path = tmp_path / "tone.wav"
    write_wav(str(path), samples, sample_rate=16000)
    return str(path)


def test_no_command_returns_zero(capsys) -> None:
    code = main([])
    assert code == EXIT_OK
    captured = capsys.readouterr()
    assert "usage" in captured.out.lower()


def test_score_command(capsys) -> None:
    code = main(["score", "abc", "axc"])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["unit"] == "character"
    assert abs(payload["rate"] - 1 / 3) < 1e-9


def test_score_command_word_unit(capsys) -> None:
    code = main(["score", "a b c", "a b c", "--unit", "word"])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["unit"] == "word"
    assert payload["rate"] == 0.0


def test_inspect_missing_file(capsys) -> None:
    code = main(["inspect", "/no/such/file.wav"])
    assert code == EXIT_RUNTIME_ERROR
    captured = capsys.readouterr()
    assert "wavrail:" in captured.err


def test_inspect_runs(wav_path, capsys) -> None:
    code = main(["inspect", wav_path])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert "num_channels" in payload


def test_decode_command(wav_path, capsys) -> None:
    code = main(["decode", wav_path, "--vocab", "<blank>,a,b,c"])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert "tokens" in payload


def test_demo_command(capsys) -> None:
    code = main(["demo"])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert "text" in payload
    assert "cer" in payload
