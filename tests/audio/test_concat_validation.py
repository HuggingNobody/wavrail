"""concatenate validation and chaining edges."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.audio import concatenate
from wavrail.errors import AudioError


def test_empty_list_rejected():
    with pytest.raises(AudioError):
        concatenate([])


def test_multidim_segment_rejected():
    with pytest.raises(AudioError):
        concatenate([np.zeros((2, 2), dtype=np.float32)])


def test_negative_crossfade_rejected():
    with pytest.raises(AudioError):
        concatenate([np.ones(8, dtype=np.float32)], crossfade_samples=-1)


def test_crossfade_longer_than_segment_rejected():
    with pytest.raises(AudioError):
        concatenate(
            [np.ones(8, dtype=np.float32), np.ones(4, dtype=np.float32)], crossfade_samples=5
        )


def test_three_segments_chain_pairwise():
    segments = [np.ones(10, dtype=np.float32) for _ in range(3)]
    out = concatenate(segments, crossfade_samples=4)
    assert out.shape == (22,)  # 30 - 2 * 4
    np.testing.assert_allclose(out, 1.0, atol=1e-6)


def test_lists_are_accepted_as_segments():
    out = concatenate([[0.5] * 4, [0.25] * 4])
    assert out.dtype == np.float32
    assert out.shape == (8,)
