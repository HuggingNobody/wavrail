"""Character Error Rate (CER) for character-based scripts.

CER is the Levenshtein edit distance divided by the reference
length in characters. The reference is treated as a sequence of
unicode code points rather than grapheme clusters; this matches
the convention used by most open-source toolkits and is
sufficient for evaluation on a normalised corpus.
"""

from __future__ import annotations

from wavrail.metrics.edit import EditOps, edit_alignment


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Return the CER of ``hypothesis`` against ``reference``.

    The reference and hypothesis are split into unicode code
    points. An empty reference is treated as a special case
    where any non-empty hypothesis yields ``inf`` and two empty
    strings yield ``0.0``.
    """
    if reference == "" and hypothesis == "":
        return 0.0
    if reference == "":
        return float("inf")
    ref_units = list(reference)
    hyp_units = list(hypothesis)
    ops = edit_alignment(ref_units, hyp_units)
    return ops.errors / max(ops.length, 1)


def character_edit_ops(reference: str, hypothesis: str) -> EditOps:
    """Return the per-character edit operations."""
    ref_units = list(reference)
    hyp_units = list(hypothesis)
    return edit_alignment(ref_units, hyp_units)


__all__ = ["character_edit_ops", "character_error_rate"]
