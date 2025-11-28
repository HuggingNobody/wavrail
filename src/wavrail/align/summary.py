"""Human-readable summaries of forced alignments.

:class:`TokenTiming` converts an alignment's frame spans into seconds
given the frame shift of the feature pipeline (``hop_length /
sample_rate``). The records are plain frozen dataclasses so reports
and CLIs can render them without touching the alignment internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

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


def detect_pauses(
    alignment: Alignment,
    frame_shift_s: float,
    min_pause_s: float = 0.1,
) -> list[tuple[float, float]]:
    """Return ``(start_s, end_s)`` gaps of at least ``min_pause_s``.

    A pause is the gap between the end of one token span and the
    start of the next; touching or overlapping spans produce no gap.
    Gaps shorter than ``min_pause_s`` are filtered out, which is how
    callers distinguish deliberate pauses from ordinary coarticulation.
    """
    _check_shift(frame_shift_s)
    if min_pause_s < 0.0:
        raise AlignmentError("min_pause_s must be non-negative")
    pauses: list[tuple[float, float]] = []
    for (_, previous_end), (next_start, _) in pairwise(alignment.frame_spans):
        gap_frames = next_start - previous_end
        if gap_frames <= 0:
            continue
        gap_s = gap_frames * frame_shift_s
        if gap_s >= min_pause_s:
            pauses.append((previous_end * frame_shift_s, next_start * frame_shift_s))
    return pauses


def summarize_alignment(
    alignment: Alignment,
    frame_shift_s: float,
    min_pause_s: float = 0.1,
) -> dict[str, object]:
    """Return a JSON-ready summary of ``alignment``.

    The key set is a stable contract: ``num_tokens``,
    ``total_frames``, ``duration_s``, ``mean_confidence``,
    ``min_confidence``, ``num_pauses``, and ``pauses_s``. Empty
    alignments are rejected because duration and confidence
    statistics are undefined for them.
    """
    if not alignment.tokens:
        raise AlignmentError("cannot summarize an empty alignment")
    _check_shift(frame_shift_s)
    pauses = detect_pauses(alignment, frame_shift_s, min_pause_s)
    confidences = list(alignment.confidences)
    starts = [start for start, _ in alignment.frame_spans]
    ends = [end for _, end in alignment.frame_spans]
    total_frames = max(ends) - min(starts)
    return {
        "num_tokens": len(alignment.tokens),
        "total_frames": int(total_frames),
        "duration_s": total_frames * frame_shift_s,
        "mean_confidence": float(sum(confidences) / len(confidences)),
        "min_confidence": float(min(confidences)),
        "num_pauses": len(pauses),
        "pauses_s": [[round(start, 6), round(end, 6)] for start, end in pauses],
    }


__all__ = ["TokenTiming", "detect_pauses", "summarize_alignment", "token_timings"]
