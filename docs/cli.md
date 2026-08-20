# CLI reference

The `wavrail` command exposes the following subcommands. All
subcommands support `-h/--help` and return non-zero exit codes on
failure.

## inspect

Print audio metadata without decoding the sample buffer.

```bash
wavrail inspect sample.wav
```

## synth

Write a deterministic synthetic waveform to a WAV file — offline
fixtures for tests and demos. `pulse` fills the first half of the
requested duration; `noise` honours `--seed`.

```bash
wavrail synth sine tone.wav --duration 1.0 --freq 440
wavrail synth noise noise.wav --seed 7 --encoding float32
```

## features

Extract log-mel features from a WAV file. `--output` writes the
feature matrix as `.npy`; frame geometry is configured in
milliseconds.

```bash
wavrail features sample.wav --n-mels 80 --output feat.npy
wavrail features sample.wav --frame-length-ms 25 --hop-length-ms 10
```

## vad

Report energy-VAD voiced regions for a WAV file as JSON: frame
counts, voiced ratio, and half-open frame-index regions.

```bash
wavrail vad sample.wav --threshold-db -40 --hangover 2
```

## decode

Run CTC greedy (`--beam 1`, default) or prefix-beam decoding on a
WAV file. Without an acoustic model the CLI derives an energy-based
surrogate posterior, so the output demonstrates the decoding path
rather than real recognition. `--json` emits the schema-versioned
hypothesis record.

```bash
wavrail decode sample.wav --vocab "<blank>,a,b,c" --beam 4 --json
```

## align

Force-align a transcript to a WAV file (both positional), using the
same energy surrogate posterior as `decode`. `--json` emits the
schema-versioned alignment record.

```bash
wavrail align sample.wav "abc" --vocab "<blank>,a,b,c" --json
```

## score

Compare a hypothesis against a reference (both positional).
`--unit` picks character- or word-level scoring, `--format` renders
text, schema JSON, JSONL, or Markdown, and `--output` writes to a
file instead of stdout.

```bash
wavrail score "hello world" "hello wurld" --unit word --format md
```

## train

Train the small synthetic GRU+CTC acoustic model. Requires the
optional `torch` extra.

```bash
wavrail train --epochs 5 --output ./checkpoints
```

## demo

Generate synthetic audio, extract features, and score a hypothesis.

```bash
wavrail demo
```
