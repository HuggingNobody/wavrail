"""Seeded property sweep over random posterior matrices.

For every random probability matrix the invariants must hold:
validation passes, entropy and confidence stay in their documented
ranges, confidence equals the row max, and temperature rescaling
preserves both the row argmax and the normalisation.
"""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.ctc import (
    frame_confidence,
    frame_entropy,
    rescale_temperature,
    validate_posteriors,
)


@pytest.mark.parametrize("seed", range(6))
def test_invariants_hold_on_random_matrices(seed: int) -> None:
    rng = np.random.default_rng(seed)
    frames = int(rng.integers(2, 12))
    classes = int(rng.integers(2, 7))
    rows = rng.random((frames, classes)) + 1e-3
    p = rows / rows.sum(axis=1, keepdims=True)

    validate_posteriors(p)  # must not raise
    entropy = frame_entropy(p)
    confidence = frame_confidence(p)
    assert entropy.shape == (frames,)
    assert confidence.shape == (frames,)
    assert np.all(entropy >= 0.0) and np.all(entropy <= 1.0)
    np.testing.assert_allclose(confidence, p.max(axis=1), atol=1e-6)

    for temperature in (0.5, 1.0, 4.0):
        out = rescale_temperature(np.log(p), temperature)
        assert np.all(out.argmax(axis=1) == p.argmax(axis=1))
        np.testing.assert_allclose(np.exp(out.astype(np.float64)).sum(axis=1), 1.0, atol=1e-5)
