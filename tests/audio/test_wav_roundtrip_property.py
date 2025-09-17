"""Seeded property sweep over the WAV write/read roundtrip.

Each case draws a random buffer length and random content from a
fixed seed, writes it through every supported encoding, and checks
that decoding recovers the samples within the codec quantisation
step. Float32 is lossless; the PCM codecs tolerate at most one full
quantisation step because the encoders round or truncate to the
nearest representable integer.
"""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.audio import read_wav, write_wav

ENCODINGS = ("pcm16", "pcm32", "float32")
MAX_ABS_ERROR = {
    "pcm16": 2.0 / 32767.0 + 1e-7,
    "pcm32": 2.0 / 2147483647.0 + 1e-7,
    "float32": 0.0,
}


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_random_buffer_roundtrip(tmp_path, encoding, seed):
    rng = np.random.default_rng(1000 + seed)
    num_samples = int(rng.integers(1, 640))
    samples = rng.uniform(-0.95, 0.95, size=num_samples).astype(np.float32)
    path = tmp_path / f"roundtrip-{encoding}-{seed}.wav"
    write_wav(path, samples, sample_rate=16000, encoding=encoding)
    decoded, meta = read_wav(path)
    assert meta.num_samples == num_samples
    assert meta.sample_rate == 16000
    assert decoded.dtype == np.float32
    if encoding == "float32":
        assert np.array_equal(decoded, samples)
    else:
        assert float(np.max(np.abs(decoded - samples))) <= MAX_ABS_ERROR[encoding]


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_single_sample_roundtrip(tmp_path, encoding):
    samples = np.array([0.25], dtype=np.float32)
    path = tmp_path / f"single-{encoding}.wav"
    write_wav(path, samples, sample_rate=8000, encoding=encoding)
    decoded, meta = read_wav(path)
    assert meta.num_samples == 1
    assert decoded[0] == pytest.approx(0.25, abs=MAX_ABS_ERROR[encoding] + 1e-7)
