"""Thin CLI dispatcher with exit-code conventions.

Exit codes:

- ``0`` success
- ``2`` usage error (bad arguments, unknown subcommand)
- ``3`` runtime error (file I/O, decoder configuration, etc.)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from wavrail import __version__

EXIT_OK = 0
EXIT_USAGE_ERROR = 2
EXIT_RUNTIME_ERROR = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wavrail",
        description="Streaming speech recognition toolkit.",
    )
    parser.add_argument("--version", action="version", version=f"wavrail {__version__}")
    sub = parser.add_subparsers(dest="command")
    _add_inspect(sub)
    _add_synth(sub)
    _add_features(sub)
    _add_decode(sub)
    _add_align(sub)
    _add_score(sub)
    _add_train(sub)
    _add_demo(sub)
    return parser


def _add_inspect(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("inspect", help="Read a WAV header and print metadata.")
    parser.add_argument("path", help="path to a WAV file")


def _add_features(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("features", help="Extract log-mel features from a WAV file.")
    parser.add_argument("path", help="path to a WAV file")
    parser.add_argument("--n-mels", type=int, default=40, help="number of mel filters (default 40)")
    parser.add_argument("--frame-length-ms", type=float, default=25.0, help="frame length in ms")
    parser.add_argument("--hop-length-ms", type=float, default=10.0, help="hop length in ms")
    parser.add_argument("--output", help="optional path to write features as .npy")


def _add_decode(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("decode", help="Run greedy or beam search over extracted features.")
    parser.add_argument("path", help="path to a WAV file")
    parser.add_argument(
        "--vocab",
        default="<blank>,a,b,c,d,e",
        help="comma-separated token labels",
    )
    parser.add_argument("--beam", type=int, default=1, help="beam size (1 = greedy)")
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the schema-versioned hypothesis record",
    )


def _add_align(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("align", help="Run forced CTC alignment of audio to a transcript.")
    parser.add_argument("path", help="path to a WAV file")
    parser.add_argument("transcript", help="transcript text")
    parser.add_argument(
        "--vocab",
        default="<blank>,a,b,c,d,e",
        help="comma-separated token labels",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the schema-versioned alignment record",
    )


def _add_score(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("score", help="Score a hypothesis against a reference.")
    parser.add_argument("reference", help="reference text")
    parser.add_argument("hypothesis", help="hypothesis text")
    parser.add_argument("--unit", choices=["character", "word"], default="character")
    parser.add_argument(
        "--format",
        choices=["text", "json", "jsonl", "md"],
        default="text",
        help="report format: text summary, schema JSON, JSONL record, Markdown",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="write the report to this path instead of stdout ('-' = stdout)",
    )


def _add_train(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("train", help="Train the synthetic acoustic model.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--save-checkpoint",
        default=None,
        help="write a resumable checkpoint to this path after training",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="resume training from this checkpoint file",
    )
    parser.add_argument(
        "--vocab",
        default="<blank>,a,b,c,d",
        help="comma-separated token labels",
    )


def _add_demo(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("demo", help="Run the end-to-end demo on synthetic audio.")
    parser.add_argument("--output", default="-", help="output JSON path, '-' for stdout")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_OK
    try:
        handler = _DISPATCH.get(args.command)
        if handler is None:
            parser.print_help()
            return EXIT_USAGE_ERROR
        handler(args)
    except FileNotFoundError as exc:
        print(f"wavrail: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    except Exception as exc:
        print(f"wavrail: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    return EXIT_OK


def _run_inspect(args: argparse.Namespace) -> None:
    from wavrail.audio import inspect_wav

    info = inspect_wav(args.path)
    payload = {
        "sample_rate": info.sample_rate,
        "num_channels": info.num_channels,
        "sample_width": info.sample_width,
        "duration_s": info.duration_s,
        "num_samples": info.num_samples,
    }
    print(json.dumps(payload, indent=2))


def _add_synth(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("synth", help="Write a synthetic waveform to a WAV file.")
    parser.add_argument(
        "kind",
        choices=["sine", "chirp", "pulse", "noise", "silence"],
        help="waveform generator to use",
    )
    parser.add_argument("output", help="destination WAV path")
    parser.add_argument("--duration", type=float, default=1.0, help="duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=16000, help="output sample rate")
    parser.add_argument("--freq", type=float, default=440.0, help="sine frequency / chirp start")
    parser.add_argument("--end-freq", type=float, default=880.0, help="chirp end frequency")
    parser.add_argument("--amplitude", type=float, default=0.5, help="peak amplitude in [0, 1]")
    parser.add_argument("--seed", type=int, default=0, help="noise generator seed")
    parser.add_argument(
        "--encoding",
        choices=["pcm16", "pcm32", "float32"],
        default="pcm16",
        help="WAV encoding (default pcm16)",
    )


def _run_synth(args: argparse.Namespace) -> None:
    from wavrail.audio import (
        synthesize_chirp,
        synthesize_noise,
        synthesize_pulse,
        synthesize_silence,
        synthesize_sine,
        write_wav,
    )

    if args.kind == "sine":
        samples = synthesize_sine(
            args.freq, args.duration, args.sample_rate, amplitude=args.amplitude
        )
    elif args.kind == "chirp":
        samples = synthesize_chirp(
            args.freq, args.end_freq, args.duration, args.sample_rate, amplitude=args.amplitude
        )
    elif args.kind == "pulse":
        # the pulse fills the first half of the requested duration
        samples = synthesize_pulse(
            0.0, args.duration / 2.0, args.duration, args.sample_rate, amplitude=args.amplitude
        )
    elif args.kind == "noise":
        samples = synthesize_noise(
            args.duration, args.sample_rate, seed=args.seed, amplitude=args.amplitude
        )
    else:
        samples = synthesize_silence(args.duration, args.sample_rate)
    write_wav(args.output, samples, args.sample_rate, encoding=args.encoding)
    payload = {
        "kind": args.kind,
        "path": str(args.output),
        "num_samples": int(samples.size),
        "encoding": args.encoding,
    }
    print(json.dumps(payload))


def _run_features(args: argparse.Namespace) -> None:
    from wavrail.audio import read_wav
    from wavrail.features.frames import frame_signal
    from wavrail.features.mel import mel_filterbank
    from wavrail.features.spectrum import power_spectrum
    from wavrail.types import FrameConfig, MelConfig

    samples, meta = read_wav(args.path)
    config = FrameConfig(
        frame_length=round(meta.sample_rate * args.frame_length_ms / 1000.0),
        hop_length=round(meta.sample_rate * args.hop_length_ms / 1000.0),
    )
    frames = frame_signal(samples, config)
    spectrum = power_spectrum(frames, fft_size=config.frame_length)
    mel_config = MelConfig(
        sample_rate=meta.sample_rate,
        num_mels=args.n_mels,
        fft_size=config.frame_length,
    )
    fbanks = mel_filterbank(mel_config)
    features = spectrum @ fbanks.T
    features = np.log(np.maximum(features, 1e-10))
    print(
        json.dumps(
            {
                "shape": list(features.shape),
                "n_mels": args.n_mels,
            }
        )
    )
    if args.output:
        np.save(args.output, features)


def _run_decode(args: argparse.Namespace) -> None:
    from wavrail.audio import read_wav
    from wavrail.ctc import PrefixBeamSearch, greedy_decode
    from wavrail.features.frames import frame_signal
    from wavrail.types import FrameConfig
    from wavrail.vocab import Vocab

    samples, _meta = read_wav(args.path)
    vocab = Vocab.from_labels(args.vocab.split(","))
    config = FrameConfig(frame_length=400, hop_length=160)
    frames = frame_signal(samples, config)
    if args.beam <= 1:
        # Use the per-frame energy as a trivial "log posterior" surrogate
        # so the demo runs without a real acoustic model.
        energy = np.log(np.maximum(frames**2, 1e-10))
        posteriors = energy[:, : vocab.size]
        posteriors = posteriors - posteriors.max(axis=1, keepdims=True)
        hyp = greedy_decode(posteriors, vocab)
    else:
        energy = np.log(np.maximum(frames**2, 1e-10))
        posteriors = energy[:, : vocab.size]
        posteriors = posteriors - posteriors.max(axis=1, keepdims=True)
        beam = PrefixBeamSearch(vocab=vocab, beam=args.beam)
        hyp = beam.decode(posteriors)
    if args.json:
        from wavrail.report import hypothesis_to_dict

        print(json.dumps(hypothesis_to_dict(hyp), ensure_ascii=False))
    else:
        print(json.dumps({"tokens": list(hyp.tokens), "text": hyp.text}))


def _run_align(args: argparse.Namespace) -> None:
    from wavrail.align import align_ctc
    from wavrail.audio import read_wav
    from wavrail.features.frames import frame_signal
    from wavrail.types import FrameConfig
    from wavrail.vocab import Vocab

    samples, _meta = read_wav(args.path)
    vocab = Vocab.from_labels(args.vocab.split(","))
    config = FrameConfig(frame_length=400, hop_length=160)
    frames = frame_signal(samples, config)
    energy = np.log(np.maximum(frames**2, 1e-10))
    posteriors = energy[:, : vocab.size]
    posteriors = posteriors - posteriors.max(axis=1, keepdims=True)
    alignment = align_ctc(posteriors, args.transcript, vocab)
    if args.json:
        from wavrail.report import align_to_dict

        print(json.dumps(align_to_dict(alignment), ensure_ascii=False))
    else:
        print(
            json.dumps(
                {
                    "tokens": list(alignment.tokens),
                    "frame_spans": [list(span) for span in alignment.frame_spans],
                }
            )
        )


def _run_score(args: argparse.Namespace) -> None:
    from wavrail.metrics import aggregate_score, character_error_rate, word_error_rate

    if args.format == "text":
        if args.unit == "character":
            rate = character_error_rate(args.reference, args.hypothesis)
        else:
            rate = word_error_rate(args.reference, args.hypothesis)
        _emit(args, json.dumps({"unit": args.unit, "rate": rate}))
        return
    from wavrail.report import score_report_to_dict, score_report_to_markdown, to_jsonl

    corpus = aggregate_score(unit=args.unit)
    corpus.add("cli", args.reference, args.hypothesis)
    payload = score_report_to_dict(corpus, title="cli-score")
    if args.format == "json":
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    elif args.format == "jsonl":
        text = to_jsonl([payload])
    else:
        text = score_report_to_markdown(payload)
    _emit(args, text)


def _emit(args: argparse.Namespace, text: str) -> None:
    """Print ``text`` or write it to ``args.output`` when set.

    A single trailing newline is guaranteed for both destinations,
    and missing parent directories are created so ``--output``
    works on fresh paths.
    """
    body = text if text.endswith("\n") else text + "\n"
    output = getattr(args, "output", "-")
    if output and output != "-":
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    else:
        sys.stdout.write(body)


def _run_train(args: argparse.Namespace) -> None:
    from wavrail.model import TrainingConfig, build_synthetic_dataset, train_synthetic
    from wavrail.vocab import Vocab

    vocab = Vocab.from_labels(args.vocab.split(","))
    config = TrainingConfig(
        n_epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
    )

    def factory(seed: int) -> tuple[list, list]:
        examples = build_synthetic_dataset(vocab, n_examples=4, duration_s=0.2, seed=seed)
        return examples[:3], examples[3:]

    report = train_synthetic(
        config,
        factory,
        vocab=vocab,
        resume_from=args.resume,
        save_to=args.save_checkpoint,
    )
    print(json.dumps(report.to_dict()))


def _run_demo(args: argparse.Namespace) -> None:
    from wavrail.audio import synthesize_formant_stack, write_wav
    from wavrail.ctc import greedy_decode
    from wavrail.features.frames import frame_signal
    from wavrail.metrics import character_error_rate
    from wavrail.types import FrameConfig
    from wavrail.vocab import Vocab

    text = "abc"
    vocab = Vocab.from_labels(["<blank>", "a", "b", "c"])
    audio = synthesize_formant_stack(
        formants_hz=[(220.0, 0.5), (440.0, 0.3), (660.0, 0.2)],
        duration_s=0.6,
        sample_rate=16000,
    )
    config = FrameConfig(frame_length=400, hop_length=160)
    frames = frame_signal(audio, config)
    energy = np.log(np.maximum(frames**2, 1e-10))[:, : vocab.size]
    posteriors = energy - energy.max(axis=1, keepdims=True)
    hyp = greedy_decode(posteriors, vocab)
    report = {
        "text": text,
        "hyp": hyp.text,
        "cer": character_error_rate(text, hyp.text),
        "audio_seconds": audio.shape[0] / 16000,
    }
    if args.output != "-":
        write_wav("/tmp/wavrail_demo.wav", audio.astype(np.float32), sample_rate=16000)
    if args.output == "-":
        print(json.dumps(report))
    else:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)


_DISPATCH = {
    "inspect": _run_inspect,
    "synth": _run_synth,
    "features": _run_features,
    "decode": _run_decode,
    "align": _run_align,
    "score": _run_score,
    "train": _run_train,
    "demo": _run_demo,
}


if __name__ == "__main__":
    raise SystemExit(main())
