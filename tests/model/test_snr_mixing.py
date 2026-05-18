"""Noise mixing in the synthetic builder: SNR accuracy and determinism."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.errors import ModelError
from wavrail.model import build_synthetic_dataset
from wavrail.model.synthetic import _MIX_HEADROOM
from wavrail.vocab import Vocab

pytestmark = pytest.mark.model

VOCAB = Vocab.from_labels(["<blank>", "a", "b", "c"])


def test_mix_hits_target_snr():
    clean = build_synthetic_dataset(VOCAB, n_examples=2, duration_s=0.2, seed=3)
    noisy = build_synthetic_dataset(VOCAB, n_examples=2, duration_s=0.2, seed=3, noise_snr_db=10.0)
    for clean_ex, noisy_ex in zip(clean, noisy, strict=True):
        peak = float(np.abs(clean_ex.audio.astype(np.float64)).max())
        scaled = clean_ex.audio.astype(np.float64) * (_MIX_HEADROOM / peak)
        noise = noisy_ex.audio.astype(np.float64) - scaled
        measured = 10.0 * np.log10(float(np.mean(scaled**2)) / float(np.mean(noise**2)))
        assert measured == pytest.approx(10.0, abs=0.3)
        assert float(np.abs(noisy_ex.audio).max()) <= 1.0


def test_mixing_is_deterministic_per_seed():
    a = build_synthetic_dataset(VOCAB, n_examples=2, duration_s=0.2, seed=5, noise_snr_db=6.0)
    b = build_synthetic_dataset(VOCAB, n_examples=2, duration_s=0.2, seed=5, noise_snr_db=6.0)
    np.testing.assert_array_equal(a[0].audio, b[0].audio)


def test_none_leaves_audio_unchanged():
    plain = build_synthetic_dataset(VOCAB, n_examples=2, duration_s=0.2, seed=5)
    again = build_synthetic_dataset(VOCAB, n_examples=2, duration_s=0.2, seed=5)
    np.testing.assert_array_equal(plain[0].audio, again[0].audio)


def test_rejects_non_finite_snr():
    with pytest.raises(ModelError):
        build_synthetic_dataset(VOCAB, n_examples=1, duration_s=0.2, noise_snr_db=float("nan"))
