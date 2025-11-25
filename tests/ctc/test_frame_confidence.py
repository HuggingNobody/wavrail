"""frame_confidence: row maximum semantics and bounds."""

from __future__ import annotations

import numpy as np

from wavrail.ctc import frame_confidence


def test_confidence_is_row_max():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    confidence = frame_confidence(p)
    assert confidence.dtype == np.float32
    np.testing.assert_allclose(confidence, [0.7, 0.8], atol=1e-6)


def test_one_hot_confidence_is_one():
    np.testing.assert_allclose(frame_confidence(np.eye(3)), 1.0, atol=1e-6)


def test_confidence_bounded_by_inverse_vocab_size():
    rng = np.random.default_rng(4)
    rows = rng.random((20, 6))
    p = rows / rows.sum(axis=1, keepdims=True)
    confidence = frame_confidence(p)
    assert np.all(confidence >= 1.0 / 6.0 - 1e-6)
    assert np.all(confidence <= 1.0)


def test_confidence_anticorrelates_with_entropy():
    from wavrail.ctc import frame_entropy

    peaked = np.array([[0.97, 0.01, 0.01, 0.01]])
    flat = np.full((1, 4), 0.25)
    assert float(frame_confidence(peaked)[0]) > float(frame_confidence(flat)[0])
    assert float(frame_entropy(peaked)[0]) < float(frame_entropy(flat)[0])
