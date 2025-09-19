"""Autocorrelation pitch estimation for a single frame.

The estimator is deliberately simple: raw autocorrelation over the
lag range implied by ``freq_range``, the lag with the highest
correlation wins, and the correlation normalised by the frame energy
is reported as ``confidence``. It is intended for voiced synthetic
fixtures, not as a production pitch tracker; frames whose best
correlation is weak (noise, silence) come back with low confidence
so callers can treat them as unvoiced.

Lag resolution bounds accuracy: at ``sample_rate`` Hz a winning lag
``L`` yields ``sample_rate / L``, so the relative error of the
estimate is at most about ``1 / L``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavrail.errors import FeatureError


@dataclass(frozen=True)
class PitchEstimate:
    """Result of a single-frame pitch estimate.

    Attributes
    ----------
    frequency_hz:
        Estimated fundamental frequency; ``0.0`` when the frame
        carries no measurable periodicity.
    confidence:
        Normalised autocorrelation at the winning lag. Values close
        to ``1.0`` indicate a strongly periodic frame.
    """

    frequency_hz: float
    confidence: float


def estimate_pitch_acf(
    frame: np.ndarray,
    sample_rate: int,
    freq_range: tuple[float, float] = (60.0, 400.0),
) -> PitchEstimate:
    """Estimate the dominant periodicity of ``frame`` via autocorrelation."""
    if sample_rate <= 0:
        raise FeatureError("sample_rate must be positive")
    low, high = freq_range
    if low <= 0.0 or high <= low:
        raise FeatureError("freq_range must satisfy 0 < low < high")
    samples = np.asarray(frame, dtype=np.float64).reshape(-1)
    lag_min = max(1, int(np.floor(sample_rate / high)))
    lag_max = int(np.ceil(sample_rate / low))
    if samples.size <= lag_max + 1:
        raise FeatureError(f"frame of {samples.size} samples is too short for lags up to {lag_max}")
    energy = float(np.dot(samples, samples))
    if energy == 0.0:
        return PitchEstimate(frequency_hz=0.0, confidence=0.0)
    lags = np.arange(lag_min, lag_max + 1)
    correlations = np.array(
        [float(np.dot(samples[: samples.size - lag], samples[lag:])) for lag in lags]
    )
    best = int(np.argmax(correlations))
    confidence = float(correlations[best] / energy)
    if confidence <= 0.0:
        return PitchEstimate(frequency_hz=0.0, confidence=0.0)
    return PitchEstimate(frequency_hz=float(sample_rate / lags[best]), confidence=confidence)


__all__ = ["PitchEstimate", "estimate_pitch_acf"]
