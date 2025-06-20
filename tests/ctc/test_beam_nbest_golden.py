"""Golden n-best scores for a fixed four-frame fixture.

The vocabulary is (blank, a, b) and the log-posteriors are the
log-softmax of the logits below. The pinned confidences equal
``exp(score / 4)`` where ``score`` is the exact CTC prefix
probability; the values were cross-checked against explicit
enumeration of all ``3 ** 4`` alignment paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.ctc import PrefixBeamSearch
from wavrail.vocab import Vocab

LOGITS = np.array(
    [
        [2.0, 1.0, 0.0],
        [0.0, 3.0, 0.5],
        [1.0, 1.0, 2.0],
        [0.2, 0.4, 3.0],
    ]
)


def _log_posteriors() -> np.ndarray:
    return LOGITS - np.log(np.exp(LOGITS).sum(axis=1, keepdims=True))


def test_golden_top_three_labels_and_confidences() -> None:
    vocab = Vocab.from_labels(["<blank>", "a", "b"])
    decoder = PrefixBeamSearch(vocab=vocab, beam=64, pruning_threshold=-1000.0)
    nbest = decoder.decode_nbest(_log_posteriors(), 3)
    assert [h.tokens for h in nbest] == [(1, 2), (2, 1, 2), (2,)]
    assert [h.text for h in nbest] == ["ab", "bab", "b"]
    assert nbest[0].confidence == pytest.approx(0.932495967, abs=1e-9)
    assert nbest[1].confidence == pytest.approx(0.537548461, abs=1e-9)
    assert nbest[2].confidence == pytest.approx(0.476092531, abs=1e-9)
