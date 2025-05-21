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


def _add_align(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("align", help="Run forced CTC alignment of audio to a transcript.")
    parser.add_argument("path", help="path to a WAV file")
    parser.add_argument("transcript", help="transcript text")
    parser.add_argument(
        "--vocab",
        default="<blank>,a,b,c,d,e",
        help="comma-separated token labels",
    )


def _add_score(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("score", help="Score a hypothesis against a reference.")
    parser.add_argument("reference", help="reference text")
    parser.add_argument("hypothesis", help="hypothesis text")
    parser.add_argument("--unit", choices=["character", "word"], default="character")


def _add_train(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("train", help="Train the synthetic acoustic model.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
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
    print(json.dumps(info, indent=2))


def _run_features(args: argparse.Namespace) -> None:
    from wavrail.audio import read_wav
    from wavrail.features.frames import frame_signal
    from wavrail.features.mel import mel_filterbank
    from wavrail.features.spectrum import power_spectrum
    from wavrail.types import FrameConfig, MelConfig

    audio = read_wav(args.path)
    config = FrameConfig(
        frame_length=round(audio.sample_rate * args.frame_length_ms / 1000.0),
        hop_length=round(audio.sample_rate * args.hop_length_ms / 1000.0),
    )
    frames = frame_signal(audio.samples, config)
    spectrum = power_spectrum(frames, fft_size=config.frame_length)
    mel_config = MelConfig(
        sample_rate=audio.sample_rate,
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

    audio = read_wav(args.path)
    vocab = Vocab.from_labels(args.vocab.split(","))
    config = FrameConfig(frame_length=400, hop_length=160)
    frames = frame_signal(audio.samples, config)
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
    print(json.dumps({"tokens": list(hyp.tokens), "text": hyp.text}))


def _run_align(args: argparse.Namespace) -> None:
    from wavrail.align import align_ctc
    from wavrail.audio import read_wav
    from wavrail.features.frames import frame_signal
    from wavrail.types import FrameConfig
    from wavrail.vocab import Vocab

    audio = read_wav(args.path)
    vocab = Vocab.from_labels(args.vocab.split(","))
    config = FrameConfig(frame_length=400, hop_length=160)
    frames = frame_signal(audio.samples, config)
    energy = np.log(np.maximum(frames**2, 1e-10))
    posteriors = energy[:, : vocab.size]
    posteriors = posteriors - posteriors.max(axis=1, keepdims=True)
    alignment = align_ctc(posteriors, args.transcript, vocab)
    print(
        json.dumps(
            {
                "tokens": list(alignment.tokens),
                "frame_spans": [list(span) for span in alignment.frame_spans],
            }
        )
    )


def _run_score(args: argparse.Namespace) -> None:
    from wavrail.metrics import character_error_rate, word_error_rate

    if args.unit == "character":
        rate = character_error_rate(args.reference, args.hypothesis)
    else:
        rate = word_error_rate(args.reference, args.hypothesis)
    print(json.dumps({"unit": args.unit, "rate": rate}))


def _run_train(args: argparse.Namespace) -> None:
    from wavrail.model import TrainingConfig, build_synthetic_dataset, train_synthetic
    from wavrail.vocab import Vocab

    vocab = Vocab.from_labels(args.vocab.split(","))
    config = TrainingConfig(
        n_epochs=args.epochs,
        hidden_dim=args.hidden_dim,
    )

    def factory(seed: int) -> tuple[list, list]:
        examples = build_synthetic_dataset(vocab, n_examples=4, duration_s=0.2, seed=seed)
        return examples[:3], examples[3:]

    report = train_synthetic(config, factory)
    print(
        json.dumps(
            {
                "initial_loss": report.initial_loss,
                "final_loss": report.final_loss,
                "loss_decreased": report.loss_decreased(),
                "epoch_losses": report.epoch_losses,
            }
        )
    )


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
    "features": _run_features,
    "decode": _run_decode,
    "align": _run_align,
    "score": _run_score,
    "train": _run_train,
    "demo": _run_demo,
}


if __name__ == "__main__":
    raise SystemExit(main())
