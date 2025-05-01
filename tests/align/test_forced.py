"""Behaviour tests for forced CTC alignment."""

from __future__ import annotations

import numpy as np

from wavrail.align import align_ctc
from wavrail.vocab import Vocab


def _vocab() -> Vocab:
    return Vocab.from_labels(["<blank>", "a", "b", "c"])


def test_alignment_covers_all_frames() -> None:
    posteriors = np.array(
        [
            [0.0, -5.0, -5.0, -5.0],
            [-5.0, 0.0, -5.0, -5.0],
            [-5.0, 0.0, -5.0, -5.0],
            [-5.0, -5.0, 0.0, -5.0],
            [-5.0, -5.0, 0.0, -5.0],
            [0.0, -5.0, -5.0, -5.0],
        ],
        dtype=np.float64,
    )
    alignment = align_ctc(posteriors, [1, 2], _vocab())
    assert alignment.tokens == (1, 2)
    assert len(alignment.frame_spans) == 2
    assert all(end > start for start, end in alignment.frame_spans)


def test_alignment_is_monotone() -> None:
    posteriors = np.array(
        [
            [0.0, -5.0, -5.0, -5.0],
            [-5.0, 0.0, -5.0, -5.0],
            [-5.0, 0.0, -5.0, -5.0],
            [0.0, -5.0, -5.0, -5.0],
            [-5.0, -5.0, 0.0, -5.0],
            [-5.0, -5.0, 0.0, -5.0],
            [0.0, -5.0, -5.0, -5.0],
        ],
        dtype=np.float64,
    )
    alignment = align_ctc(posteriors, [1, 2], _vocab())
    starts = [span[0] for span in alignment.frame_spans]
    assert starts == sorted(starts)
    assert all(
        alignment.frame_spans[i][1] <= alignment.frame_spans[i + 1][1]
        for i in range(len(alignment.frame_spans) - 1)
    )


def test_alignment_string_target() -> None:
    posteriors = np.array(
        [
            [0.0, -5.0, -5.0, -5.0],
            [-5.0, 0.0, -5.0, -5.0],
            [-5.0, 0.0, -5.0, -5.0],
            [0.0, -5.0, -5.0, -5.0],
            [-5.0, -5.0, 0.0, -5.0],
            [-5.0, -5.0, 0.0, -5.0],
            [0.0, -5.0, -5.0, -5.0],
        ],
        dtype=np.float64,
    )
    alignment = align_ctc(posteriors, "ab", _vocab())
    assert alignment.tokens == (1, 2)


def test_alignment_rejects_blank_in_target() -> None:
    posteriors = np.zeros((3, 4), dtype=np.float64)
    try:
        align_ctc(posteriors, [0, 1], _vocab())
    except Exception as exc:
        assert "blank" in str(exc).lower()
    else:
        raise AssertionError("expected an exception")
