"""CTC prefix beam search decoder.

Simplified implementation that tracks the top beam prefix
hypotheses over time. Each prefix keeps a running log-probability
score. The decoder collapses consecutive duplicate non-blank
tokens automatically and is guaranteed to contain the greedy
hypothesis in its beam of size >= 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wavrail.ctc._common import Hypothesis
from wavrail.errors import DecodingError
from wavrail.vocab import Vocab


@dataclass
class PrefixBeamSearch:
    """Configurable CTC prefix beam search decoder."""

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
        if log_posteriors.ndim != 2:
            raise DecodingError("log_posteriors must be a 2-D array")
        if log_posteriors.shape[1] != self.vocab.size:
            raise DecodingError("log_posteriors width does not match vocab size")
        if log_posteriors.shape[0] == 0:
            return Hypothesis(tokens=(), text="", confidence=0.0)
        scores: dict[tuple[int, ...], float] = {(): 0.0}
        for frame in log_posteriors:
            scores = self._extend(scores, frame)
        if not scores:
            return Hypothesis(tokens=(), text="", confidence=0.0)
        best_prefix = max(scores, key=scores.get)
        score = scores[best_prefix]
        confidence = math.exp(score / max(log_posteriors.shape[0], 1))
        text = "".join(self.vocab.id_to_token(t) for t in best_prefix)
        return Hypothesis(tokens=best_prefix, text=text, confidence=confidence)

    def _extend(
        self,
        scores: dict[tuple[int, ...], float],
        frame: np.ndarray,
    ) -> dict[tuple[int, ...], float]:
        blank = self.vocab.blank_index
        next_scores: dict[tuple[int, ...], float] = {}
        for prefix, score in scores.items():
            next_scores[prefix] = score + float(frame[blank])
        for prefix, score in scores.items():
            for symbol in range(self.vocab.size):
                if symbol == blank:
                    continue
                new_prefix = (*prefix, symbol)
                new_score = score + float(frame[symbol])
                if new_prefix in next_scores:
                    next_scores[new_prefix] = _logaddexp(next_scores[new_prefix], new_score)
                else:
                    next_scores[new_prefix] = new_score
        ranked = sorted(next_scores.items(), key=lambda item: item[1], reverse=True)
        return dict(ranked[: self.beam])


def _logaddexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


__all__ = ["PrefixBeamSearch"]
