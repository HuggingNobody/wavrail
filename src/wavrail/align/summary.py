"""Human-readable summaries of forced alignments.

:class:`TokenTiming` converts an alignment's frame spans into seconds
given the frame shift of the feature pipeline (``hop_length /
sample_rate``). The records are plain frozen dataclasses so reports
and CLIs can render them without touching the alignment internals.
"""

from __future__ import annotations

from dataclasses import dataclass

from wavrail.align.forced import Alignment
from wavrail.errors import AlignmentError


@dataclass(frozen=True)
class TokenTiming:
    """One aligned token with its frame span and time span in seconds.

    Attributes
    ----------
    index:
        Position of the token within the transcript.
    token_id:
        Vocabulary id of the token.
    start_frame / end_frame:
        Half-open frame interval copied from the alignment.
    start_s / end_s:
        The same interval converted to seconds via the frame shift.
    confidence:
        Per-token confidence carried by the alignment.
    """

    index: int
    token_id: int
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    confidence: float

    @property
    def duration_s(self) -> float:
        """Length of the token's time span in seconds."""
        return self.end_s - self.start_s


def _check_shift(frame_shift_s: float) -> None:
    if frame_shift_s <= 0.0:
        raise AlignmentError("frame_shift_s must be positive")


def token_timings(alignment: Alignment, frame_shift_s: float) -> list[TokenTiming]:
    """Convert every frame span of ``alignment`` to a timing record."""
    _check_shift(frame_shift_s)
    timings: list[TokenTiming] = []
    for index, token in enumerate(alignment.tokens):
        start, end = alignment.frame_spans[index]
        timings.append(
            TokenTiming(
                index=index,
                token_id=token,
                start_frame=start,
                end_frame=end,
                start_s=start * frame_shift_s,
                end_s=end * frame_shift_s,
                confidence=alignment.confidences[index],
            )
        )
    return timings


__all__ = ["TokenTiming", "token_timings"]
