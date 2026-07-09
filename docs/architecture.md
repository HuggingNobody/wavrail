# Architecture

wavrail is structured as a small set of cooperating subpackages with
minimal cross-module coupling.

```
wavrail/
  audio/        WAV I/O, format validation, synthetic generators, editing helpers
  features/     Framing, windows, mel filterbanks, log-mel, CMVN, MFCC, spectral
                descriptors, pitch, energy VAD
  vocab/        Token<->id vocabularies with unicode safety and file persistence
  ctc/          Greedy, prefix-beam, frame-synchronous emission, posterior tools
  align/        Forced CTC alignment with timing and pause summaries
  metrics/      Text normalization, CER, WER, edit alignment, confusion counts
  stream/       Chunked pipelines, ring buffers, push-based framing, latency
  model/        Optional torch extra: GRU encoder + CTC head, training loop,
                early stopping, inference helpers
  report/       Versioned JSON schemas, JSONL / Markdown exporters
  cli/          Subcommand entry points built on argparse
```

## Layering rules

- `audio` depends only on NumPy and the standard library.
- `features` consumes audio arrays and produces 2-D feature matrices.
- `vocab` is dependency-free and exports a frozen mapping.
- `ctc` consumes posteriors and a vocabulary to produce hypotheses.
- `align` consumes posteriors, a transcript, and a vocabulary to
  produce frame-accurate token spans.
- `metrics` consumes strings (no dependency on the decoder).
- `stream` orchestrates `features` and `ctc` with bounded buffers.
- `model` is optional and only imported under `wavrail.model`.
- `cli` is a thin wrapper around the public library API.
- `report` provides versioned serialization for downstream tools.

## Data flow

```
WAV bytes --[audio]--> float32 samples --[features]--> log-mel
   log-mel --[model?]--> posteriors --[ctc]--> hypothesis
   reference + hypothesis --[metrics]--> CER / WER report
```

The streaming pipeline composes `features` and `ctc` into a chunked
loop with bounded lookahead and resumable state.
