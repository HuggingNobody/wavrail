"""Shared helpers for the CTC decoders.

Provides:
- :class:`Hypothesis` dataclass returned by every decoder.
- :func:`collapse_blanks` helper that removes consecutive blanks
  and repeated non-blank tokens.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Hypothesis:
    """A decoded CTC hypothesis.

    Attributes
    ----------
    tokens:
        Sequence of token ids (excluding the blank token).
    text:
        The token sequence rendered as Unicode text via the
        supplied vocabulary.
    confidence:
        Geometric mean of the per-frame posterior probability of
        the chosen symbol. The value is bounded to ``[0, 1]``;
        callers should treat it as a relative ordering signal
        rather than a calibrated probability.
    """

    tokens: tuple[int, ...]
    text: str
    confidence: float
    frame_indices: tuple[int, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.tokens)


def collapse_blanks(sequence: Sequence[int], blank_index: int) -> list[int]:
    """Remove consecutive blanks and repeated non-blank tokens.

    This is the standard CTC collapse: walk the frame-by-frame
    argmax and emit a token whenever the current id differs from
    the previous non-blank id.
    """
    out: list[int] = []
    previous = -1
    for token in sequence:
        if token == blank_index:
            previous = -1
            continue
        if token == previous:
            continue
        out.append(int(token))
        previous = int(token)
    return out


def argmax_per_frame(posteriors: np.ndarray) -> np.ndarray:
    """Return the per-frame argmax of a posterior matrix."""
    if posteriors.ndim != 2:
        raise ValueError("posteriors must be a 2-D array")
    return np.argmax(posteriors, axis=1).astype(np.int64)


__all__ = ["Hypothesis", "argmax_per_frame", "collapse_blanks"]
