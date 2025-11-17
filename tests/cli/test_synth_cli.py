"""Tests for the synth subcommand."""

from __future__ import annotations

import json

from wavrail.audio import read_wav
from wavrail.cli import EXIT_OK, EXIT_RUNTIME_ERROR, main


def test_synth_sine_writes_wav(tmp_path) -> None:
    out = tmp_path / "tone.wav"
    code = main(["synth", "sine", str(out), "--duration", "0.1", "--freq", "300"])
    assert code == EXIT_OK
    _, meta = read_wav(str(out))
    assert meta.sample_rate == 16000
    assert meta.num_samples == 1600


def test_synth_noise_is_seeded(tmp_path) -> None:
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    main(["synth", "noise", str(a), "--duration", "0.05", "--seed", "7"])
    main(["synth", "noise", str(b), "--duration", "0.05", "--seed", "7"])
    first, _ = read_wav(str(a))
    second, _ = read_wav(str(b))
    assert (first == second).all()


def test_synth_reports_payload(tmp_path, capsys) -> None:
    out = tmp_path / "s.wav"
    main(["synth", "silence", str(out), "--duration", "0.25"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "silence"
    assert payload["num_samples"] == 4000
    assert payload["encoding"] == "pcm16"


def test_synth_pulse_roundtrip_amplitude(tmp_path) -> None:
    out = tmp_path / "p.wav"
    main(
        [
            "synth",
            "pulse",
            str(out),
            "--duration",
            "0.2",
            "--amplitude",
            "0.8",
            "--encoding",
            "float32",
        ]
    )
    samples, meta = read_wav(str(out))
    assert meta.num_samples == 3200
    assert float(samples[:1600].max()) > 0.79  # first half carries the pulse


def test_synth_unwritable_path_exits_runtime_error(tmp_path, capsys) -> None:
    code = main(["synth", "sine", str(tmp_path / "missing" / "x.wav")])
    assert code == EXIT_RUNTIME_ERROR
