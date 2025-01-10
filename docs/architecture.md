# Architecture

wavrail is structured as a small set of cooperating subpackages with
minimal cross-module coupling.

```
wavrail/
  audio/        WAV I/O, format validation, synthetic generators
  features/     Framing, windows, mel filterbanks, log-mel, CMVN
  vocab/        Token<->id vocabularies with unicode safety
  ctc/          Greedy, prefix-beam, frame-synchronous emission, forced alignment
  metrics/      Text normalization, CER, WER, edit alignment
  streaming/    Chunked pipelines, ring buffers, latency accounting
  model/        Optional torch extra: GRU encoder + CTC head
  report/       Versioned JSON schemas, JSONL / Markdown exporters
  cli/          Subcommand entry points built on argparse
```

## Layering rules

- `audio` depends only on NumPy and the standard library.
- `features` consumes audio arrays and produces 2-D feature matrices.
- `vocab` is dependency-free and exports a frozen mapping.
- `ctc` consumes posteriors and a vocabulary to produce hypotheses
  or alignments.
- `metrics` consumes strings (no dependency on the decoder).
- `streaming` orchestrates `features` and `ctc` with bounded buffers.
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
