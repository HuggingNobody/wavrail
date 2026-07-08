"""Determinism and structure of the 04_audio_quality example."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "04_audio_quality" / "main.py"


def _load():
    spec = importlib.util.spec_from_file_location("example_04_audio_quality", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analyze_is_deterministic():
    example = _load()
    samples = example.build_fixture()
    assert example.analyze(samples) == example.analyze(samples)


def test_payload_structure():
    example = _load()
    payload = example.analyze(example.build_fixture())
    assert payload["num_samples"] == 10400  # 0.65 s at 16 kHz
    assert payload["trimmed_samples"] < payload["num_samples"]
    assert payload["voiced_frames"] > 0
    assert payload["num_frames"] > 0
    assert len(payload["silence_regions"]) >= 2
    assert all(start < end for start, end in payload["silence_regions"])


def test_main_prints_json(capsys):
    example = _load()
    assert example.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert "mean_centroid_hz" in payload
    assert "mean_flatness" in payload
