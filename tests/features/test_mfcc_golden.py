"""Golden MFCC coefficients on a fixed synthetic log-mel input.

The expected matrix was produced by the shipped pipeline on the
seeded input below and is pinned to an absolute tolerance of 1e-4,
the float32 golden tolerance documented in the numerical tolerance
policy. Regenerating the golden requires a deliberate change to the
DCT or lifter stage.
"""

from __future__ import annotations

import numpy as np

from wavrail.features import mfcc_from_logmel

GOLDEN = np.array(
    [
        [-24.3851909637, 1.342520833, 0.7849795818, -2.875535965],
        [-21.5039424896, -0.2705336213, -2.5044410229, 0.8057806492],
        [-22.3158569336, -0.4078922868, 0.4401402175, -0.743717134],
    ]
)


def _fixture_logmel() -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.standard_normal((3, 8)) * 2.0 - 8.0).astype(np.float32)


def test_mfcc_golden_values():
    out = mfcc_from_logmel(_fixture_logmel(), num_coeffs=4, lifter=0)
    np.testing.assert_allclose(out, GOLDEN, atol=1e-4)


def test_dc_coefficient_matches_scaled_row_mean():
    logmel = _fixture_logmel()
    out = mfcc_from_logmel(logmel, num_coeffs=1, lifter=0)
    expected = logmel.astype(np.float64).mean(axis=1) * np.sqrt(8.0)
    np.testing.assert_allclose(out[:, 0], expected, rtol=1e-5)
