"""max_symbols_per_frame caps the per-frame extension fan-out."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.ctc import PrefixBeamSearch, greedy_decode
from wavrail.errors import DecodingError
from wavrail.vocab import Vocab


def _vocab() -> Vocab:
    return Vocab.from_labels(["<blank>", "a", "b", "c"])


def test_cap_one_follows_dominant_symbols() -> None:
    lp = np.array(
        [
            [-8.0, 0.0, -8.0, -8.0],
            [-8.0, -8.0, 0.0, -8.0],
            [-8.0, -8.0, -8.0, 0.0],
        ]
    )
    decoder = PrefixBeamSearch(vocab=_vocab(), beam=4, max_symbols_per_frame=1)
    assert decoder.decode(lp).tokens == greedy_decode(lp, _vocab()).tokens == (1, 2, 3)


def test_cap_excludes_low_ranked_extensions() -> None:
    # 'a' outranks 'b' in both frames; with a cap of one no prefix
    # containing 'b' can ever be created
    lp = np.array([[-9.0, -0.5, -1.0, -9.0], [-9.0, -0.5, -1.0, -9.0]])
    # beam 16 keeps every reachable prefix (10 at T=2, V=4), so the
    # capped token set must be a strict subset of the uncapped one
    capped = PrefixBeamSearch(
        vocab=_vocab(), beam=16, pruning_threshold=-1000.0, max_symbols_per_frame=1
    )
    uncapped = PrefixBeamSearch(vocab=_vocab(), beam=16, pruning_threshold=-1000.0)
    capped_tokens = {h.tokens for h in capped.decode_nbest(lp, 16)}
    uncapped_tokens = {h.tokens for h in uncapped.decode_nbest(lp, 16)}
    assert (2,) in uncapped_tokens
    assert (2,) not in capped_tokens
    assert capped_tokens < uncapped_tokens


def test_rejects_non_positive_cap() -> None:
    with pytest.raises(DecodingError, match="max_symbols_per_frame"):
        PrefixBeamSearch(vocab=_vocab(), max_symbols_per_frame=0)
