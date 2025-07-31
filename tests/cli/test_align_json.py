"""align --json emits the schema-versioned alignment record."""

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


def test_align_json_flag_emits_schema_record(wav_path, capsys) -> None:
    code = main(["align", str(wav_path), "ab", "--json"])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "alignment"
    assert payload["schema_version"] == "1"
    assert payload["tokens"] == [1, 2]
    assert len(payload["frame_spans"]) == 2
    assert len(payload["confidences"]) == 2


def test_align_default_output_unchanged(wav_path, capsys) -> None:
    code = main(["align", str(wav_path), "ab"])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"tokens", "frame_spans"}
