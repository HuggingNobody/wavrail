"""Decode synthetic posteriors and report CER."""

from __future__ import annotations

import json

import numpy as np

from wavrail.ctc import PrefixBeamSearch, greedy_decode
from wavrail.metrics import character_error_rate
from wavrail.vocab import Vocab


def _build_posteriors(text: str, vocab: Vocab, *, frames_per_token: int = 2) -> np.ndarray:
    frames: list[np.ndarray] = []
    for ch in text:
        token_id = vocab.token_to_id(ch)
        row = np.full(vocab.size, -5.0)
        row[token_id] = 0.0
        for _ in range(frames_per_token):
            frames.append(row.copy())
        row_blank = np.full(vocab.size, -5.0)
        row_blank[vocab.blank_index] = 0.0
        frames.append(row_blank)
    return np.stack(frames)


def main() -> int:
    vocab = Vocab.from_labels(["<blank>", "a", "b", "c", "d"])
    text = "abcd"
    posteriors = _build_posteriors(text, vocab)
    greedy = greedy_decode(posteriors, vocab)
    beam = PrefixBeamSearch(vocab=vocab, beam=4).decode(posteriors)
    payload = {
        "reference": text,
        "greedy_text": greedy.text,
        "greedy_cer": character_error_rate(text, greedy.text),
        "beam_text": beam.text,
        "beam_cer": character_error_rate(text, beam.text),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
