# 01_features_golden

Generate a deterministic synthetic formant sweep, extract log-mel
features, and compare the output statistics against the expected
golden shape. This exercises the full feature pipeline
(frame blocking, power spectrum, mel filterbank, log compression)
on CPU without any acoustic-model weights.

## Run

```bash
python -m examples.01_features_golden.main
```

The example prints a JSON report describing the log-mel shape and
basic statistics, and writes the feature matrix to
`/tmp/wavrail_example_01_log_mel.npy` for inspection.

## Notes

- All randomness is seeded by the `noise_seed` argument.
- The synthetic audio is NOT real speech; it exists only to make
  the feature pipeline fully reproducible.