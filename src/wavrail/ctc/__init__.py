"""Connectionist Temporal Classification (CTC) decoders.

This subpackage provides greedy decoding, prefix beam search,
frame-synchronous emission, and forced alignment. All decoders
are implemented in pure NumPy so that wavrail does not require a
deep-learning runtime outside the optional torch extra.

The decoders operate on 2-D arrays of shape
``(num_frames, vocab_size)``. ``posteriors`` may be either
probabilities or log-probabilities; the maximum-argmax step is
identical and downstream log handling is explicit.
"""

from wavrail.ctc._common import Hypothesis, collapse_blanks
from wavrail.ctc.beam import PrefixBeamSearch
from wavrail.ctc.greedy import greedy_decode
from wavrail.ctc.posteriors import frame_entropy, validate_posteriors

__all__ = [
    "Hypothesis",
    "PrefixBeamSearch",
    "collapse_blanks",
    "frame_entropy",
    "greedy_decode",
    "validate_posteriors",
]
