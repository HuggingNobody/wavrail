# Examples

This directory contains runnable offline examples. Each example ships
with its own README and produces deterministic outputs.

| Example | Description | Requires torch |
| ------- | ----------- | -------------- |
| `01_features_golden` | Extract log-mel features from synthetic audio and compare against a golden reference. | No |
| `02_decode_synthetic` | Decode synthetic posteriors with the greedy and prefix-beam decoders and report CER. | No |
| `03_train_decode_score` | Train a small GRU+CTC acoustic model on synthetic data, decode, and score. | Yes |
| `04_audio_quality` | Detect silence regions, trim, run energy VAD, pitch, and spectral descriptors on synthetic audio. | No |

Run an example with `python -m examples.01_features_golden.main`
from the repository root.
