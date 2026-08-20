"""Tests for the vad subcommand."""

from __future__ import annotations

import json

from wavrail.audio import synthesize_pulse, synthesize_silence, write_wav
from wavrail.cli import EXIT_OK, EXIT_RUNTIME_ERROR, main


def _write(tmp_path, samples, name: str = "v.wav") -> str:
    path = tmp_path / name
    write_wav(str(path), samples, 16000)
    return str(path)


def test_vad_reports_voiced_regions(tmp_path, capsys) -> None:
    path = _write(tmp_path, synthesize_pulse(0.05, 0.05, 0.2, 16000, amplitude=0.5))
    code = main(["vad", path])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["num_frames"] == 18  # 400-sample frames with a 160 hop
    assert payload["voiced_frames"] > 0
    assert len(payload["regions"]) >= 1
    for start, end in payload["regions"]:
        assert 0 <= start < end <= 18


def test_vad_on_silence_has_no_regions(tmp_path, capsys) -> None:
    path = _write(tmp_path, synthesize_silence(0.2, 16000))
    main(["vad", path])
    payload = json.loads(capsys.readouterr().out)
    assert payload["voiced_frames"] == 0
    assert payload["regions"] == []
    assert payload["voiced_ratio"] == 0.0


def test_vad_missing_file_exits_runtime_error(tmp_path, capsys) -> None:
    code = main(["vad", str(tmp_path / "absent.wav")])
    assert code == EXIT_RUNTIME_ERROR
