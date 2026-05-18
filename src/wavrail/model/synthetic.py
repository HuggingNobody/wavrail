"""Synthetic dataset builder for the acoustic model demo.

This module generates :class:`SyntheticExample` objects whose
``features`` are log-mel spectrograms of synthetic formant
trajectories and whose ``tokens`` are the corresponding target
token id sequences. The output is **explicitly documented as
not real speech**: it exists only to exercise the training and
inference pipeline end-to-end on a CPU-only environment without
network access or pretrained weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wavrail.audio import synthesize_formant_stack
from wavrail.errors import ModelError
from wavrail.features.frames import frame_signal
from wavrail.features.mel import mel_filterbank
from wavrail.features.spectrum import power_spectrum
from wavrail.types import FrameConfig, MelConfig
from wavrail.vocab import Vocab


@dataclass
class SyntheticExample:
    """A single synthetic training example."""

    name: str
    audio: np.ndarray
    sample_rate: int
    features: np.ndarray
    tokens: list[int]
    text: str


def _log_mel(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_length: int = 400,
    hop_length: int = 160,
    n_mels: int = 40,
) -> np.ndarray:
    config = FrameConfig(frame_length=frame_length, hop_length=hop_length)
    frames = frame_signal(audio, config)
    spectrum = power_spectrum(frames, fft_size=frame_length)
    mel_config = MelConfig(
        sample_rate=sample_rate,
        num_mels=n_mels,
        fft_size=frame_length,
    )
    fbanks = mel_filterbank(mel_config)
    mel_energy = spectrum @ fbanks.T
    mel_energy = np.maximum(mel_energy, 1e-10)
    return np.log(mel_energy).astype(np.float32)


def _formant_for_token(token: str, base: float) -> list[tuple[float, float]]:
    """Map a token to a deterministic formant trajectory."""
    h = sum(ord(c) for c in token)
    f1 = base + (h % 200)
    f2 = f1 + 400 + (h % 150)
    f3 = f2 + 600 + (h % 100)
    return [(f1, 0.5), (f2, 0.3), (f3, 0.2)]


_MIX_HEADROOM = 0.7


def _mix_noise(audio: np.ndarray, rng: np.random.Generator, snr_db: float) -> np.ndarray:
    """Scale ``audio`` to headroom and add uniform noise at ``snr_db``.

    The clean signal is peak-normalised to :data:`_MIX_HEADROOM`
    before mixing so the added noise has room under full scale; the
    result is still clipped to ``[-1, 1]`` as a safety net. SNR is
    defined on power: ``10 * log10(signal_power / noise_power)``.
    Silent input stays silent because its SNR is undefined.
    """
    clean = audio.astype(np.float64)
    peak = float(np.abs(clean).max())
    if peak > 0.0:
        clean = clean * (_MIX_HEADROOM / peak)
    signal_power = float(np.mean(clean * clean))
    if signal_power <= 0.0:
        return clean.astype(np.float32)
    noise = rng.uniform(-1.0, 1.0, size=clean.shape[0])
    noise_power = float(np.mean(noise * noise))
    target_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = noise * math.sqrt(target_power / noise_power)
    return np.clip(clean + noise, -1.0, 1.0).astype(np.float32)


def build_synthetic_dataset(
    vocab: Vocab,
    *,
    n_examples: int = 16,
    duration_s: float = 0.5,
    sample_rate: int = 16000,
    seed: int = 0,
    noise_snr_db: float | None = None,
) -> list[SyntheticExample]:
    """Build a deterministic synthetic dataset of ``n_examples`` items.

    With ``noise_snr_db`` set, every example's audio is peak-scaled to
    0.7 headroom and mixed with seeded uniform noise at the requested
    signal-to-noise ratio — still synthetic formant stacks, not real
    noisy speech.
    """
    if n_examples < 1:
        raise ModelError("n_examples must be >= 1")
    if duration_s <= 0:
        raise ModelError("duration_s must be > 0")
    if noise_snr_db is not None and not math.isfinite(noise_snr_db):
        raise ModelError("noise_snr_db must be finite")
    rng = np.random.default_rng(seed)
    base = 220.0
    examples: list[SyntheticExample] = []
    token_pool = [label for label in vocab.labels if not vocab.is_blank(vocab.token_to_id(label))]
    if not token_pool:
        raise ModelError("vocab has no non-blank tokens")
    for index in range(n_examples):
        n_tokens = 1 + int(rng.integers(0, 3))
        tokens: list[int] = []
        text_chars: list[str] = []
        for _ in range(n_tokens):
            token = token_pool[int(rng.integers(0, len(token_pool)))]
            tokens.append(vocab.token_to_id(token))
            text_chars.append(token)
        text = "".join(text_chars)
        sample_count = max(1, round(duration_s * sample_rate))
        pieces: list[np.ndarray] = []
        for token_id in tokens:
            formants = _formant_for_token(vocab.id_to_token(token_id), base + token_id * 40)
            piece_len = sample_count // max(n_tokens, 1)
            wave = synthesize_formant_stack(
                formants_hz=formants,
                duration_s=piece_len / sample_rate,
                sample_rate=sample_rate,
                noise_amplitude=0.0,
                noise_seed=seed,
            )
            pieces.append(wave)
        audio = np.concatenate(pieces)
        if audio.shape[0] < sample_count:
            audio = np.concatenate([audio, np.zeros(sample_count - audio.shape[0])])
        else:
            audio = audio[:sample_count]
        if noise_snr_db is not None:
            audio = _mix_noise(audio, rng, noise_snr_db)
        features = _log_mel(audio, sample_rate)
        examples.append(
            SyntheticExample(
                name=f"utt_{index:03d}",
                audio=audio.astype(np.float32),
                sample_rate=sample_rate,
                features=features,
                tokens=tokens,
                text=text,
            )
        )
    return examples


__all__ = ["SyntheticExample", "build_synthetic_dataset"]
