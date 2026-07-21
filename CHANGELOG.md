# Changelog

All notable changes to this project will be documented in this file. The
format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] - 2026-09-26

### Added

- Initial package scaffold (`pyproject.toml`, src layout, hatchling build).
- Shared exception hierarchy (`wavrail.errors`).
- Shared dataclasses and type aliases (`wavrail.types`).
- WAV codecs (PCM16/PCM32/Float32), resampling, downmix, quality flags,
  and seeded synthetic waveform generators.
- Feature stack: framing, windows, pre-emphasis, power spectrum, mel
  filterbanks, log-mel, streaming CMVN, deltas, spectral descriptors,
  frame energy, pitch, energy VAD, DCT-II, liftering, MFCC, and
  zero-crossing rate.
- Audio editing helpers: silence trimming, DC-offset removal, segment
  concatenation with crossfade, and silence-region detection.
- CTC decoding: greedy, prefix beam search with n-best output, posterior
  validation, entropy and confidence summaries, and temperature rescaling.
- Frame-synchronous emission decoder with serializable streaming state.
- Forced CTC alignment with token timings, pause detection, and JSON
  summaries.
- Metrics: CER/WER, edit alignments, text normalization, confusion
  counts, corpus slicing, and rate summaries.
- Vocab persistence (versioned JSON and tokens.txt) and `Vocab.from_text`.
- Streaming: ring buffer, chunked pipeline with latency accounting,
  push-based framer, and streaming spectrogram with state serde.
- Model extras (torch): GRU+CTC encoder, synthetic dataset builder with
  optional SNR noise mixing, training loop with warmup, decay, and early
  stopping, checkpoints, and inference helpers.
- Reports: versioned JSON schemas, JSONL and Markdown exporters.
- CLI subcommands: inspect, synth, features, decode, align, score,
  train, demo.
- Four offline examples under `examples/`.

## [0.1.0] - 2026-09-25

### Added

- First tagged release. Establishes the public module layout and exposes
  the version constant `wavrail.__version__`.

[Unreleased]: https://github.com/StarWorkshop/wavrail/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/StarWorkshop/wavrail/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/StarWorkshop/wavrail/releases/tag/v0.1.0