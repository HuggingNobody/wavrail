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

Extract log-mel features to stdout as a `.npy` file or a JSON array.

```bash
wavrail features sample.wav --num-mels 80 --format npy > feat.npy
```

## decode

Run CTC greedy or prefix-beam decoding on posterior text or a
checkpoint-emitted posterior file.

```bash
wavrail decode --beam 4 sample.posteriors.json
```

## align

Run forced alignment between a transcript and a posterior stream.

```bash
wavrail align --transcript "hello" sample.posteriors.json
```

## score

Compare a hypothesis file or text against a reference.

```bash
wavrail score --reference "hello" --hypothesis "hello world"
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
