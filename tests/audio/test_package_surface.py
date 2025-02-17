"""Smoke test verifying the full audio public surface imports."""

from __future__ import annotations

import wavrail.audio as audio


def test_audio_public_surface_exports_expected_symbols() -> None:
    expected = {
        "clipping_detected",
        "downmix_to_mono",
        "inspect_wav",
        "read_wav",
        "resample",
        "silence_ratio",
        "synthesize_chirp",
        "synthesize_formant_stack",
        "synthesize_noise",
        "synthesize_pulse",
        "synthesize_sine",
        "write_wav",
    }
    assert expected.issubset(set(audio.__all__))


def test_clipping_and_silence_helpers_round_trip() -> None:
    sig = audio.synthesize_sine(440.0, 0.1, 16000, amplitude=1.0)
    assert audio.clipping_detected(sig)
    silence = audio.synthesize_pulse(0.0, 0.001, 0.1, 16000)
    assert audio.silence_ratio(silence) > 0.95
