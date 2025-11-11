"""Seeded property: trimming removes exactly the edge silence regions.

The fixture is ``head`` silent samples, a body drawn uniformly from
``[0.3, 0.8]`` (always loud at -60 dBFS), and ``tail`` silent
samples. The detected regions must be precisely the head and tail
runs, and the trimmed length must equal the body length.
"""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.audio import detect_silence_regions, trim_silence


@pytest.mark.parametrize("seed", range(5))
def test_trim_matches_edge_regions(seed: int) -> None:
    rng = np.random.default_rng(seed)
    head = int(rng.integers(0, 400))
    tail = int(rng.integers(0, 400))
    body_len = int(rng.integers(200, 800))
    body = rng.uniform(0.3, 0.8, size=body_len).astype(np.float32)
    samples = np.concatenate(
        [np.zeros(head, dtype=np.float32), body, np.zeros(tail, dtype=np.float32)]
    )
    regions = detect_silence_regions(samples, threshold_db=-60.0)
    expected = []
    if head:
        expected.append((0, head))
    if tail:
        expected.append((head + body_len, int(samples.size)))
    assert regions == expected
    assert trim_silence(samples, threshold_db=-60.0).size == body_len


def test_body_with_interior_quiet_sample_is_preserved():
    body = np.full(64, 0.5, dtype=np.float32)
    body[32] = 0.0  # interior quiet sample: regions see it, trimming does not
    samples = np.concatenate([np.zeros(8, dtype=np.float32), body])
    regions = detect_silence_regions(samples, threshold_db=-60.0)
    assert regions == [(0, 8), (40, 41)]
    assert trim_silence(samples, threshold_db=-60.0).size == 64
