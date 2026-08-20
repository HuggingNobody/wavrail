"""Tests for the features subcommand output kinds."""

from __future__ import annotations

import json

import numpy as np

from wavrail.audio import synthesize_sine, write_wav
from wavrail.cli import EXIT_OK, main


def _wav(tmp_path) -> str:
    samples = synthesize_sine(440.0, 0.25, 16000)
    path = tmp_path / "tone.wav"
    write_wav(str(path), samples, 16000)
    return str(path)


def test_logmel_is_default(tmp_path, capsys) -> None:
    code = main(["features", _wav(tmp_path), "--n-mels", "24"])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "logmel"
    assert payload["n_mels"] == 24
    assert payload["shape"][1] == 24


def test_mfcc_kind_uses_num_coeffs(tmp_path, capsys) -> None:
    code = main(["features", _wav(tmp_path), "--kind", "mfcc", "--num-coeffs", "8"])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "mfcc"
    assert payload["num_coeffs"] == 8
    assert payload["shape"][1] == 8


def test_output_writes_npy(tmp_path, capsys) -> None:
    out = tmp_path / "feat.npy"
    main(
        [
            "features",
            _wav(tmp_path),
            "--kind",
            "mfcc",
            "--num-coeffs",
            "5",
            "--output",
            str(out),
        ]
    )
    features = np.load(str(out))
    assert features.shape[1] == 5
