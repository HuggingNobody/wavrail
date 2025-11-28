"""CTC forced alignment.

Given a posterior matrix ``P`` of shape ``(T, V)`` and a known
target transcript ``Y = (y_1, ..., y_L)``, the forced alignment
finds the best frame-by-frame mapping that produces ``Y`` under
the CTC topology. The implementation expands the target with
blanks to length ``2L+1`` and runs the standard CTC Viterbi
algorithm with forbidden transitions.

The output is an :class:`Alignment` carrying the per-token start
and end frames and a per-token confidence derived from the
posterior mass accumulated while emitting that token.
"""

from wavrail.align.forced import (
    Alignment,
    align_ctc,
)
from wavrail.align.summary import TokenTiming, detect_pauses, token_timings

__all__ = ["Alignment", "TokenTiming", "align_ctc", "detect_pauses", "token_timings"]
