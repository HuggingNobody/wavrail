"""concatenate: lengths, crossfade ramp, and copies."""

from __future__ import annotations

import numpy as np

from wavrail.audio import concatenate


def test_plain_concatenation():
    a = np.ones(100, dtype=np.float32)
    b = np.full(50, 0.5, dtype=np.float32)
    out = concatenate([a, b])
    assert out.shape == (150,)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[:100], 1.0)
    np.testing.assert_allclose(out[100:], 0.5)


def test_crossfade_shortens_output():
    a = np.ones(100, dtype=np.float32)
    out = concatenate([a, a.copy()], crossfade_samples=20)
    assert out.shape == (180,)
    np.testing.assert_allclose(out, 1.0, atol=1e-6)  # equal levels fade into themselves


def test_crossfade_is_linear_ramp():
    a = np.ones(64, dtype=np.float32)
    b = np.zeros(64, dtype=np.float32)
    out = concatenate([a, b], crossfade_samples=10)
    assert out.shape == (118,)
    expected = np.linspace(1.0, 0.0, 10, dtype=np.float32)
    np.testing.assert_allclose(out[54:64], expected, atol=1e-6)
    np.testing.assert_allclose(out[64:], 0.0, atol=1e-7)


def test_single_segment_returns_copy():
    a = np.ones(10, dtype=np.float32)
    out = concatenate([a])
    assert out is not a
    np.testing.assert_array_equal(out, a)


def test_inputs_are_not_mutated():
    a = np.ones(32, dtype=np.float32)
    b = np.full(32, 0.25, dtype=np.float32)
    a_before, b_before = a.copy(), b.copy()
    concatenate([a, b], crossfade_samples=8)
    np.testing.assert_array_equal(a, a_before)
    np.testing.assert_array_equal(b, b_before)
