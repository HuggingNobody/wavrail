"""Shared fixtures for streaming pipeline tests.

``sign_extractor`` maps each raw frame to a peaked three-symbol
log-posterior row and ``square_wave`` drives it deterministically,
so emitted token patterns can be asserted exactly. Both are
synthetic test fixtures, not audio features.
"""

from __future__ import annotations

import numpy as np


def sign_extractor(frames: np.ndarray) -> np.ndarray:
    """Return ``(num_frames, 3)`` rows picking 'a' or 'b' by frame mean sign."""
    means = frames.mean(axis=1)
    rows = np.full((frames.shape[0], 3), -5.0)
    rows[means > 0, 1] = 5.0
    rows[means <= 0, 2] = 5.0
    return rows


def square_wave(segment_samples: int = 800, segments: int = 3) -> np.ndarray:
    """Alternating ``+0.8`` / ``-0.8`` plateaus as ``float32``."""
    parts = [
        np.full(segment_samples, 0.8 if i % 2 == 0 else -0.8, dtype=np.float32)
        for i in range(segments)
    ]
    return np.concatenate(parts)
