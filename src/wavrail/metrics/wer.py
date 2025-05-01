"""Word Error Rate (WER) for space-delimited languages.

Words are produced by ``str.split()`` after collapsing internal
whitespace. Empty reference strings are treated the same way as
in :func:`wavrail.metrics.cer.character_error_rate`.
"""

from __future__ import annotations

from wavrail.metrics.edit import EditOps, edit_alignment


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Return the WER of ``hypothesis`` against ``reference``."""
    if reference == "" and hypothesis == "":
        return 0.0
    if reference == "":
        return float("inf")
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    ops = edit_alignment(ref_words, hyp_words)
    return ops.errors / max(ops.length, 1)


def word_edit_ops(reference: str, hypothesis: str) -> EditOps:
    """Return the per-word edit operations."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    return edit_alignment(ref_words, hyp_words)


__all__ = ["word_edit_ops", "word_error_rate"]
