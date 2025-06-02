"""Greedy decoder edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.ctc import greedy_decode
from wavrail.errors import DecodingError
from wavrail.vocab import Vocab


def test_empty_input() -> None:
    vocab = Vocab.from_labels(["<blank>", "a", "b"])
    hyp = greedy_decode(np.zeros((0, 3), dtype=np.float64), vocab)
    assert hyp.tokens == ()


def test_single_blank_frame() -> None:
    vocab = Vocab.from_labels(["<blank>", "a", "b"])
    hyp = greedy_decode(np.array([[0.0, -5.0, -5.0]], dtype=np.float64), vocab)
    assert hyp.tokens == ()


def test_rejects_wrong_width() -> None:
    vocab = Vocab.from_labels(["<blank>", "a", "b"])
    with pytest.raises(DecodingError):
        greedy_decode(np.zeros((4, 5), dtype=np.float64), vocab)
