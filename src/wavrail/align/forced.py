"""CTC forced alignment via Viterbi over an expanded label topology.

The expanded target is ``[b, y_1, b, y_2, ..., b, y_L, b]``. At
each frame we consider three transitions:

1. Stay on the same label (consume a frame, no new token).
2. Advance to the next label (consume a frame, emit token).
3. If the next label is not blank and not the same as the
   current label, advance by two (skip the blank between
   identical non-blank tokens which CTC forbids without the
   blank in between).

The result is a per-token ``[start, end]`` frame interval
together with a per-token confidence derived from the posterior
mass accumulated while that token was active.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wavrail.errors import AlignmentError
from wavrail.vocab import Vocab


@dataclass
class Alignment:
    """Per-token frame intervals and per-token confidence.

    Attributes
    ----------
    tokens:
        The token sequence that was forced-aligned.
    frame_spans:
        ``frame_spans[i]`` is the half-open ``[start, end)``
        interval of frames during which token ``i`` was active.
        ``end`` is exclusive so the length of the span equals the
        number of frames the token was held.
    log_probabilities:
        The per-token log posterior accumulated across the span
        (sum, not mean — sum is what CTC training minimises).
    confidences:
        Geometric mean of the per-frame posterior probability for
        each token, normalised to ``[0, 1]``.
    path:
        The decoded expanded-label path of length ``T``. Each
        entry is an index into ``expanded_labels``.
    """

    tokens: tuple[int, ...]
    frame_spans: tuple[tuple[int, int], ...]
    log_probabilities: tuple[float, ...]
    confidences: tuple[float, ...]
    path: tuple[int, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.tokens)


def align_ctc(
    log_posteriors: np.ndarray,
    target: list[int] | tuple[int, ...] | str,
    vocab: Vocab,
) -> Alignment:
    """Run forced CTC alignment of ``log_posteriors`` to ``target``.

    ``target`` may be either a string of tokens or a sequence of
    token ids. Passing a string requires ``vocab.encode`` to map
    each character to a token id; passing an id sequence skips
    that step.
    """
    if log_posteriors.ndim != 2:
        raise AlignmentError("log_posteriors must be a 2-D array")
    if log_posteriors.shape[1] != vocab.size:
        raise AlignmentError("log_posteriors width does not match vocab size")
    if log_posteriors.shape[0] == 0:
        return Alignment(tokens=(), frame_spans=(), log_probabilities=(), confidences=())
    token_ids = vocab.encode(target) if isinstance(target, str) else [int(t) for t in target]
    if not token_ids:
        return Alignment(tokens=(), frame_spans=(), log_probabilities=(), confidences=())
    blank = vocab.blank_index
    if any(t == blank for t in token_ids):
        raise AlignmentError("target tokens must not include the blank id")
    expanded = _expand_with_blanks(token_ids, blank)
    path = _viterbi(log_posteriors, expanded, blank)
    spans, log_ps = _collapse_path(path, expanded)
    confidences = tuple(
        float(np.exp(lp / max(end - start, 1)))
        for (start, end), lp in zip(spans, log_ps, strict=True)
    )
    return Alignment(
        tokens=tuple(token_ids),
        frame_spans=tuple(spans),
        log_probabilities=tuple(log_ps),
        confidences=confidences,
        path=tuple(int(p) for p in path),
    )


def _expand_with_blanks(tokens: list[int], blank: int) -> list[int]:
    expanded: list[int] = [blank]
    for token in tokens:
        expanded.append(token)
        expanded.append(blank)
    return expanded


def _viterbi(
    log_posteriors: np.ndarray,
    expanded: list[int],
    blank: int,
) -> list[int]:
    time_steps, _vocab_size = log_posteriors.shape
    labels = len(expanded)
    neg_inf = -1e18
    alpha = np.full((time_steps, labels), neg_inf, dtype=np.float64)
    beta = np.full((time_steps, labels), -1, dtype=np.int64)
    alpha[0, 0] = float(log_posteriors[0, expanded[0]])
    if labels > 1:
        alpha[0, 1] = float(log_posteriors[0, expanded[1]])
        beta[0, 1] = 0
    for t in range(1, time_steps):
        prev = alpha[t - 1]
        cur = alpha[t]
        bcur = beta[t]
        for s in range(labels):
            best = prev[s]
            src = s
            if s > 0 and prev[s - 1] > best:
                best = prev[s - 1]
                src = s - 1
            if (
                s > 1
                and expanded[s] != blank
                and expanded[s - 2] != expanded[s]
                and prev[s - 2] > best
            ):
                best = prev[s - 2]
                src = s - 2
            cur[s] = best + float(log_posteriors[t, expanded[s]])
            bcur[s] = src
    path = [0] * time_steps
    last = labels - 1
    path[-1] = last
    for t in range(time_steps - 1, 0, -1):
        last = int(beta[t, last])
        path[t - 1] = last
    return path


def _collapse_path(
    path: list[int], expanded: list[int]
) -> tuple[list[tuple[int, int]], list[float]]:
    spans: list[tuple[int, int]] = []
    log_ps: list[float] = []
    current = -1
    start = 0
    log_acc = 0.0
    for t, label_idx in enumerate(path):
        token = expanded[label_idx]
        if token == current:
            continue
        if current >= 0:
            spans.append((start, t))
            log_ps.append(log_acc)
        current = token
        start = t
        log_acc = 0.0
    if current >= 0:
        spans.append((start, len(path)))
        log_ps.append(log_acc)
    return spans, log_ps


__all__ = ["Alignment", "align_ctc"]
