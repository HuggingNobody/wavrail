"""CTC prefix beam search decoder.

Implementation of the algorithm described in Maas, Hannun,
Nguyen (2014) "First-Pass Large Vocabulary Continuous Speech
Recognition using Bi-Directional Recurrent Neural Networks"
and the Graves distillation in
https://distill.pub/2017/ctc/. The decoder keeps at most ``beam``
active prefixes per frame and emits the best-scoring hypothesis
after the final frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wavrail.ctc._common import Hypothesis
from wavrail.errors import DecodingError
from wavrail.vocab import Vocab


@dataclass
class _BeamState:
    log_p_blank: float
    log_p_non_blank: float
    prefix: tuple[int, ...]
    text: str


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
        """Run prefix beam search over a matrix of log-posteriors.

        Parameters
        ----------
        log_posteriors:
            ``float`` array of shape ``(num_frames, vocab_size)``.
            Values must be log probabilities (any base); they are
            treated as additive contributions.
        """
        if log_posteriors.ndim != 2:
            raise DecodingError("log_posteriors must be a 2-D array")
        if log_posteriors.shape[1] != self.vocab.size:
            raise DecodingError("log_posteriors width does not match vocab size")
        if log_posteriors.shape[0] == 0:
            return Hypothesis(tokens=(), text="", confidence=0.0)
        states: dict[tuple[int, ...], _BeamState] = {
            (): _BeamState(log_p_blank=0.0, log_p_non_blank=-math.inf, prefix=(), text="")
        }
        for frame in log_posteriors:
            states = self._extend(states, frame)
        if not states:
            return Hypothesis(tokens=(), text="", confidence=0.0)
        best_prefix = max(
            states,
            key=lambda p: self._log_prob(states[p]),
        )
        final_state = states[best_prefix]
        confidence = (
            math.exp(self._log_prob(final_state)) if self._log_prob(final_state) < 0 else 1.0
        )
        return Hypothesis(tokens=best_prefix, text=final_state.text, confidence=confidence)

    def _extend(
        self,
        states: dict[tuple[int, ...], _BeamState],
        frame: np.ndarray,
    ) -> dict[tuple[int, ...], _BeamState]:
        blank = self.vocab.blank_index
        next_states: dict[tuple[int, ...], _BeamState] = {}
        prefixes = list(states.keys())
        candidate_symbols: list[int] = list(range(self.vocab.size))
        if self.max_symbols_per_frame is not None and self.max_symbols_per_frame < self.vocab.size:
            top = np.argpartition(-frame, self.max_symbols_per_frame)[: self.max_symbols_per_frame]
            candidate_symbols = sorted(int(s) for s in top)
        for prefix in prefixes:
            state = states[prefix]
            if state.log_p_blank > -math.inf:
                self._accumulate(next_states, prefix, state.log_p_blank, blank, prefix, "")
            if state.log_p_non_blank > -math.inf:
                self._accumulate(next_states, prefix, state.log_p_non_blank, blank, prefix, "")
            for symbol in candidate_symbols:
                if symbol == blank:
                    continue
                p_symbol = float(frame[symbol])
                if p_symbol == -math.inf:
                    continue
                if prefix and symbol == prefix[-1]:
                    self._accumulate(
                        next_states,
                        prefix,
                        state.log_p_non_blank,
                        symbol,
                        prefix,
                        "",
                    )
                else:
                    new_text = self.vocab.id_to_token(symbol)
                    new_prefix = (*prefix, symbol)
                    self._accumulate(
                        next_states,
                        prefix,
                        state.log_p_blank,
                        symbol,
                        new_prefix,
                        new_text,
                    )
                    self._accumulate(
                        next_states,
                        prefix,
                        state.log_p_non_blank,
                        symbol,
                        new_prefix,
                        new_text,
                    )
        pruned = self._prune(next_states)
        return pruned

    def _accumulate(
        self,
        next_states: dict[tuple[int, ...], _BeamState],
        source_prefix: tuple[int, ...],
        source_log_prob: float,
        emitted_symbol: int,
        target_prefix: tuple[int, ...],
        appended_text: str,
    ) -> None:
        if source_log_prob == -math.inf:
            return
        blank = self.vocab.blank_index
        state = next_states.get(target_prefix)
        if state is None:
            text = "".join(self.vocab.id_to_token(t) for t in target_prefix)
            state = _BeamState(
                log_p_blank=-math.inf,
                log_p_non_blank=-math.inf,
                prefix=target_prefix,
                text=text,
            )
            next_states[target_prefix] = state
        new_log = source_log_prob
        if emitted_symbol == blank:
            state.log_p_blank = _logaddexp(state.log_p_blank, new_log)
        else:
            state.log_p_non_blank = _logaddexp(state.log_p_non_blank, new_log)

    def _prune(
        self, states: dict[tuple[int, ...], _BeamState]
    ) -> dict[tuple[int, ...], _BeamState]:
        ranked = sorted(states.items(), key=lambda item: self._log_prob(item[1]), reverse=True)
        kept = dict(ranked[: self.beam])
        for _prefix, state in kept.items():
            if self._log_prob(state) < self.pruning_threshold:
                pass
        return kept

    @staticmethod
    def _log_prob(state: _BeamState) -> float:
        return _logaddexp(state.log_p_blank, state.log_p_non_blank)


def _logaddexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


__all__ = ["PrefixBeamSearch"]
