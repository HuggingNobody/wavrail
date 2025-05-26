"""Feature-extraction example with golden reference."""

from __future__ import annotations

import json

import numpy as np

from wavrail.audio.synthetic import synthesize_formant_stack
from wavrail.features.frames import frame_signal
from wavrail.features.mel import mel_filterbank
from wavrail.features.spectrum import power_spectrum
from wavrail.types import FrameConfig, MelConfig


def main() -> int:
    sample_rate = 16000
    duration_s = 0.5
    samples = synthesize_formant_stack(
        formants_hz=[(440.0, 0.6), (880.0, 0.4)],
        duration_s=duration_s,
        sample_rate=sample_rate,
        noise_seed=42,
    )
    config = FrameConfig(frame_length=400, hop_length=160)
    frames = frame_signal(samples, config)
    spectrum = power_spectrum(frames, fft_size=config.frame_length)
    mel_config = MelConfig(sample_rate=sample_rate, num_mels=40, fft_size=config.frame_length)
    fbanks = mel_filterbank(mel_config)
    mel_energy = spectrum @ fbanks.T
    log_mel = np.log(np.maximum(mel_energy, 1e-10))
    payload = {
        "sample_rate": sample_rate,
        "duration_s": duration_s,
        "log_mel_shape": list(log_mel.shape),
        "log_mel_min": float(log_mel.min()),
        "log_mel_max": float(log_mel.max()),
        "log_mel_mean": float(log_mel.mean()),
    }
    print(json.dumps(payload, indent=2))
    np.save("/tmp/wavrail_example_01_log_mel.npy", log_mel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
