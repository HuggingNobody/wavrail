# 03_train_decode_score

Train the small GRU+CTC acoustic model on a synthetic dataset,
then report the held-out synthetic CER. This example requires the
optional `torch` extra:

```bash
pip install wavrail[torch]
python -m examples.03_train_decode_score.main
```

## Notes

- The training set and dev set are both synthesised from random
  formant trajectories; the data is NOT real speech and any CER
  reported here should be read as a smoke test, not a benchmark.
- The example verifies that the loss decreases during training
  and that the synthetic pipeline runs end-to-end.