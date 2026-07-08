"""Audio-quality analysis example: regions, trimming, VAD, pitch, spectra.

Builds a synthetic fixture (silence, a seeded formant "voice"
segment, silence) and runs the wavrail analysis stack fully offline:
silence-region detection, head/tail trimming, energy VAD,
autocorrelation pitch on the first voiced frames, and spectral
descriptors. The printed JSON payload is deterministic.

The fixture is NOT real speech; pitch and descriptor values are only
meaningful for this synthetic signal.
"""

from __future__ import annotations

import json

import numpy as np

from wavrail.audio import (
    concatenate,
    detect_silence_regions,
    synthesize_formant_stack,
    synthesize_silence,
    trim_silence,
)
from wavrail.features import (
    detect_voice_activity,
    estimate_pitch_acf,
    frame_signal,
    hann_window,
    power_spectrum,
    spectral_centroid,
    spectral_flatness,
)
from wavrail.types import FrameConfig

SAMPLE_RATE = 16000


def build_fixture(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Silence, a voiced formant segment, then silence; fully seeded."""
    lead = synthesize_silence(0.1, sample_rate)
    voice = synthesize_formant_stack(
        formants_hz=[(220.0, 0.5), (440.0, 0.3), (660.0, 0.2)],
        duration_s=0.4,
        sample_rate=sample_rate,
        noise_amplitude=0.01,
        noise_seed=7,
    )
    tail = synthesize_silence(0.15, sample_rate)
    return concatenate([lead, voice, tail])


def analyze(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict[str, object]:
    """Run the analysis stack and return a deterministic JSON-ready payload."""
    regions = detect_silence_regions(samples, threshold_db=-50.0, min_region_samples=160)
    trimmed = trim_silence(samples, threshold_db=-50.0)
    config = FrameConfig(frame_length=400, hop_length=160, center=False)
    frames = frame_signal(trimmed, config)
    flags = detect_voice_activity(frames, threshold_db=-40.0, hangover_frames=2)
    voiced = [row for row, flag in zip(frames, flags, strict=True) if flag]
    pitches = [estimate_pitch_acf(row, sample_rate).frequency_hz for row in voiced[:5]]
    spectra = power_spectrum(frames, fft_size=512, window=hann_window(400))
    centroid = spectral_centroid(spectra, sample_rate)
    flatness = spectral_flatness(spectra, sample_rate)
    return {
        "num_samples": int(samples.size),
        "silence_regions": [[int(start), int(end)] for start, end in regions],
        "trimmed_samples": int(trimmed.size),
        "num_frames": int(frames.shape[0]),
        "voiced_frames": int(flags.sum()),
        "first_pitches_hz": [round(float(pitch), 3) for pitch in pitches],
        "mean_centroid_hz": round(float(centroid.mean()), 3),
        "mean_flatness": round(float(flatness.mean()), 6),
    }


def main() -> int:
    payload = analyze(build_fixture())
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
