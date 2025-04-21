"""CTC greedy decoder.

The greedy decoder takes the per-frame argmax of the posterior
matrix, applies the standard CTC collapse (remove consecutive
blanks and repeated non-blank tokens), and emits a
:class:`wavrail.ctc._common.Hypothesis` with a confidence
estimate derived from the geometric mean of the chosen token's
posterior probability.
"""

from __future__ import annotations

import numpy as np

from wavrail.ctc._common import Hypothesis, argmax_per_frame, collapse_blanks
from wavrail.errors import DecodingError
from wavrail.vocab import Vocab


def greedy_decode(posteriors: np.ndarray, vocab: Vocab) -> Hypothesis:
    """Decode a posterior matrix using the greedy CTC algorithm."""
    if posteriors.ndim != 2:
        raise DecodingError("posteriors must be a 2-D array")
    if posteriors.shape[1] != vocab.size:
        raise DecodingError(
            f"posterior width {posteriors.shape[1]} does not match vocab size {vocab.size}"
        )
    argmax = argmax_per_frame(posteriors)
    tokens = collapse_blanks(argmax.tolist(), vocab.blank_index)
    confidence = _confidence(posteriors, argmax)
    text = "".join(vocab.id_to_token(t) for t in tokens)
    return Hypothesis(tokens=tuple(tokens), text=text, confidence=confidence)


def _confidence(posteriors: np.ndarray, argmax: np.ndarray) -> float:
    if posteriors.shape[0] == 0:
        return 0.0
    chosen = posteriors[np.arange(posteriors.shape[0]), argmax]
    if np.any(chosen <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(chosen))))


__all__ = ["greedy_decode"]
