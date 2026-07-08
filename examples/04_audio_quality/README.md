# 04_audio_quality

Analyze a synthetic fixture with the wavrail audio-quality stack:
silence-region detection, head/tail trimming, energy VAD,
autocorrelation pitch on voiced frames, and spectral descriptors.
Everything runs offline on generated audio — no datasets, no model
weights.

## Run

```bash
python -m examples.04_audio_quality.main
```

The example prints a deterministic JSON payload: detected silence
regions, trimmed length, voiced-frame count, the first few pitch
estimates, and mean spectral centroid and flatness.

## Notes

- The fixture is silence + a seeded formant stack + silence; it is
  NOT real speech.
- Pitch estimates come from raw autocorrelation and are meaningful
  only for the synthetic voiced segment.
- Region detection uses `min_region_samples=160` so the low-level
  noise bed inside the voiced segment does not fragment it.
