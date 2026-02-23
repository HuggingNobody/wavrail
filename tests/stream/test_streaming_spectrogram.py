"""StreamingSpectrogram offline equivalence and contracts."""

from __future__ import annotations

import numpy as np
import pytest

from wavrail.errors import PipelineError
from wavrail.features import frame_signal, hann_window, power_spectrum
from wavrail.stream import StreamingSpectrogram
from wavrail.types import FrameConfig

CFG = FrameConfig(frame_length=64, hop_length=32, center=False)


def test_rows_match_offline_power_spectrum():
    signal = np.random.default_rng(5).standard_normal(1000).astype(np.float32)
    window = hann_window(64)
    spectrogram = StreamingSpectrogram(CFG, fft_size=128, window=window)
    rows = np.vstack([spectrogram.push(signal[i : i + 100]) for i in range(0, 1000, 100)])
    offline = power_spectrum(frame_signal(signal, CFG), fft_size=128, window=window)
    assert rows.shape == offline.shape
    np.testing.assert_allclose(rows, offline, rtol=1e-6, atol=1e-6)


def test_num_bins_contract():
    spectrogram = StreamingSpectrogram(CFG, fft_size=128)
    assert spectrogram.num_bins == 65
    assert spectrogram.push(np.zeros(31, dtype=np.float32)).shape == (0, 65)


def test_flush_pad_emits_tail_row():
    signal = np.random.default_rng(6).standard_normal(100).astype(np.float32)
    spectrogram = StreamingSpectrogram(CFG, fft_size=128)
    spectrogram.push(signal)  # two complete frames from 100 samples
    rows = spectrogram.flush(pad_tail=True)
    assert rows.shape == (1, 65)


def test_rejects_fft_smaller_than_frame():
    with pytest.raises(PipelineError):
        StreamingSpectrogram(CFG, fft_size=32)
    with pytest.raises(PipelineError):
        StreamingSpectrogram(CFG, fft_size=0)
