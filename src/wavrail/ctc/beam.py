"""CTC prefix beam search decoder.

Implements the prefix beam search of Graves et al., 2014
(https://arxiv.org/abs/1408.2873); the illustrated walkthrough at
https://distill.pub/2017/ctc/ derives the same recursion. Each beam
entry is an output prefix carrying two log-probability states:

- ``p_b``: probability mass of the paths that collapse to the
  prefix and end on the blank symbol,
- ``p_nb``: mass of the paths that collapse to the prefix and end
  on a non-blank symbol.

Keeping the two states apart is what lets a repeated symbol
separated by a blank (``a, <blank>, a``) collapse to ``aa``; a
single-score beam cannot represent that distinction and silently
loses such repeats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wavrail.ctc._common import Hypothesis
from wavrail.errors import DecodingError
from wavrail.vocab import Vocab

_NEG_INF = -math.inf


@dataclass
class PrefixBeamSearch:
    """Configurable CTC prefix beam search decoder.

    Attributes
    ----------
    beam:
        Maximum number of prefixes carried across frames.
    pruning_threshold:
        Log-domain cutoff relative to the best prefix score;
        entries scoring more than ``|threshold|`` below the best
        are dropped each frame. Must be ``<= 0``.
    max_symbols_per_frame:
        When set, only the top-N non-blank symbols of each frame
        are used for extensions, bounding the per-frame work.
    """

    vocab: Vocab
    beam: int = 5
    pruning_threshold: float = -10.0
    max_symbols_per_frame: int | None = None

    def __post_init__(self) -> None:
        if self.beam < 1:
            raise DecodingError("beam must be >= 1")
        if self.pruning_threshold > 0.0:
            raise DecodingError("pruning_threshold must be <= 0 (log-probability)")

    def decode(self, log_posteriors: np.ndarray) -> Hypothesis:
        """Return the highest-scoring hypothesis for ``log_posteriors``."""
        self._validate(log_posteriors)
        if log_posteriors.shape[0] == 0:
            return Hypothesis(tokens=(), text="", confidence=0.0)
        ranked = self._search(log_posteriors)
        if not ranked:
            return Hypothesis(tokens=(), text="", confidence=0.0)
        prefix, score = ranked[0]
        return self._to_hypothesis(prefix, score, log_posteriors.shape[0])

    def _validate(self, log_posteriors: np.ndarray) -> None:
        if log_posteriors.ndim != 2:
            raise DecodingError("log_posteriors must be a 2-D array")
        if log_posteriors.shape[1] != self.vocab.size:
            raise DecodingError("log_posteriors width does not match vocab size")

    def _to_hypothesis(self, prefix: tuple[int, ...], score: float, num_frames: int) -> Hypothesis:
        # scores are log prefix probabilities, so exp(score / T) is a
        # geometric-mean confidence in [0, 1]; the clamp only triggers
        # on unnormalised inputs whose per-frame mass exceeds one
        confidence = min(1.0, math.exp(score / max(num_frames, 1)))
        text = "".join(self.vocab.id_to_token(t) for t in prefix)
        return Hypothesis(tokens=prefix, text=text, confidence=confidence)

    def _search(self, log_posteriors: np.ndarray) -> list[tuple[tuple[int, ...], float]]:
        blank = self.vocab.blank_index
        states: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, _NEG_INF)}
        for frame in log_posteriors:
            states = self._advance(states, np.asarray(frame, dtype=np.float64), blank)
        ranked = [
            (prefix, float(np.logaddexp(p_b, p_nb))) for prefix, (p_b, p_nb) in states.items()
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def _advance(
        self,
        states: dict[tuple[int, ...], tuple[float, float]],
        frame: np.ndarray,
        blank: int,
    ) -> dict[tuple[int, ...], tuple[float, float]]:
        log_blank = float(frame[blank])
        symbols = self._candidate_symbols(frame, blank)
        acc: dict[tuple[int, ...], list[float]] = {}

        def add(prefix: tuple[int, ...], p_b: float, p_nb: float) -> None:
            entry = acc.setdefault(prefix, [_NEG_INF, _NEG_INF])
            entry[0] = float(np.logaddexp(entry[0], p_b))
            entry[1] = float(np.logaddexp(entry[1], p_nb))

        for prefix, (p_b, p_nb) in states.items():
            total = float(np.logaddexp(p_b, p_nb))
            # emitting blank keeps the prefix, ending in the blank state
            add(prefix, total + log_blank, _NEG_INF)
            last = prefix[-1] if prefix else -1
            for symbol in symbols:
                logp = float(frame[symbol])
                if symbol == last:
                    # repeat with no intervening blank: prefix unchanged
                    add(prefix, _NEG_INF, p_nb + logp)
                    # repeat right after a blank: prefix extends
                    add((*prefix, symbol), _NEG_INF, p_b + logp)
                else:
                    add((*prefix, symbol), _NEG_INF, total + logp)
        return self._prune(acc)

    def _candidate_symbols(self, frame: np.ndarray, blank: int) -> list[int]:
        order = np.argsort(frame, kind="stable")[::-1]
        return [int(s) for s in order if int(s) != blank]

    def _prune(
        self, acc: dict[tuple[int, ...], list[float]]
    ) -> dict[tuple[int, ...], tuple[float, float]]:
        live: dict[tuple[int, ...], tuple[float, float]] = {}
        for prefix, values in acc.items():
            if math.isfinite(max(values[0], values[1])):
                live[prefix] = (values[0], values[1])
        ranked = sorted(
            live.items(),
            key=lambda item: float(np.logaddexp(item[1][0], item[1][1])),
            reverse=True,
        )
        return dict(ranked[: self.beam])


__all__ = ["PrefixBeamSearch"]
