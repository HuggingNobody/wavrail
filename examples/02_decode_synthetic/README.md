# 02_decode_synthetic

Run the greedy decoder and the prefix beam search decoder over a
synthetic CTC posterior matrix and report the character error rate
of each. The posterior matrix encodes the reference text as a
sequence of per-token argmax spikes interleaved with blanks.

## Run

```bash
python -m examples.02_decode_synthetic.main
```

## Notes

- The synthetic posteriors are NOT model output; they are a
  hand-crafted pattern so the demo works on CPU without a trained
  acoustic model.
- Greedy and beam search should produce the same text on this
  input; both CERs should be 0.0.