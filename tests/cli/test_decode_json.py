"""decode --json emits the schema-versioned hypothesis record."""

from __future__ import annotations

import json

import numpy as np
import pytest

from wavrail.audio import synthesize_sine, write_wav
from wavrail.cli.main import EXIT_OK, main


@pytest.fixture()
def wav_path(tmp_path):
    path = tmp_path / "tone.wav"
    wave = synthesize_sine(frequency_hz=440.0, duration_s=0.2, sample_rate=16000)
    write_wav(path, wave.astype(np.float32), sample_rate=16000)
    return path


def test_decode_json_flag_emits_schema_record(wav_path, capsys) -> None:
    code = main(["decode", str(wav_path), "--json"])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "hypothesis"
    assert payload["schema_version"] == "1"
    assert "tokens" in payload
    assert 0.0 <= payload["confidence"] <= 1.0


def test_decode_json_with_beam(wav_path, capsys) -> None:
    code = main(["decode", str(wav_path), "--beam", "3", "--json"])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "hypothesis"


def test_decode_default_output_unchanged(wav_path, capsys) -> None:
    code = main(["decode", str(wav_path)])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"tokens", "text"}
