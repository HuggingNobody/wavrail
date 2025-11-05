# Usage

This guide covers the library entry points. The CLI is documented in
`docs/cli.md`.

## Reading audio

```python
import wavrail.audio as audio

samples, meta = audio.read_wav("sample.wav")
print(meta.sample_rate, meta.num_samples)
```

## Synthesizing fixtures

```python
tone = audio.synthesize_sine(frequency_hz=440.0,
                             duration_s=1.0,
                             sample_rate=16000)
```

## Feature extraction

```python
import wavrail.features as ft
from wavrail.types import FrameConfig, MelConfig

frame_config = FrameConfig(frame_length=400, hop_length=160)
mel_config = MelConfig(sample_rate=16000, num_mels=80, fft_size=512)
log_mel = ft.log_mel_spectrogram(samples, 16000, frame_config, mel_config)
```

## Spectral descriptors and VAD

```python
import wavrail.features as ft

frames = ft.frame_signal(samples, frame_config)
spectra = ft.power_spectrum(frames, fft_size=512, window=ft.hann_window(400))
centroid = ft.spectral_centroid(spectra, 16000)   # Hz, one value per frame
rolloff = ft.spectral_rolloff(spectra, 16000, percent=0.85)
flatness = ft.spectral_flatness(spectra, 16000)   # ~1 noise-like, ~0 tonal
flux = ft.spectral_flux(spectra)                  # onset strength per frame

flags = ft.detect_voice_activity(frames, threshold_db=-40.0, hangover_frames=2)
pitch = ft.estimate_pitch_acf(frames[0], 16000)   # PitchEstimate(hz, confidence)
```

## MFCC and cepstral features

```python
import wavrail.features as ft

coeffs = ft.mfcc(samples, 16000, frame_config, mel_config, num_coeffs=13)
staged = ft.mfcc_from_logmel(log_mel, num_coeffs=13, lifter=22)
weights = ft.cepstral_lifter(13, lifter=22)  # HTK-style lifter curve
zcr = ft.zero_crossing_rate(ft.frame_signal(samples, frame_config))
```

## CTC decoding

```python
import wavrail.vocab as vocab
import wavrail.ctc as ctc

v = vocab.Vocab.from_labels(["<blank>", "a", "b", "c"])
decoder = ctc.PrefixBeamSearch(vocab=v, beam=4)
hypothesis = decoder.decode(posteriors)
```

## Metrics

```python
import wavrail.metrics as metrics

wer = metrics.word_error_rate(reference="hello world", hypothesis="hello wurld")
cer = metrics.character_error_rate(reference="今天天气好", hypothesis="今天气好")
ops = metrics.edit_alignment("hello world".split(), "hello wurld".split())
print(wer, ops.substitutions, ops.insertions, ops.deletions)
```

## Streaming

```python
import wavrail.stream as stream
import wavrail.vocab as vocab

v = vocab.Vocab.from_labels(["<blank>", "a", "b", "c"])
pipe = stream.StreamingPipeline(sample_rate=16000, vocab=v)
for chunk in chunks:  # float32 mono audio, e.g. 100 ms at a time
    token_ids = pipe.feed_samples(chunk)  # bounded-lookahead emissions
    print(token_ids)
hypothesis = pipe.finalize()  # flush the tail after the last chunk
print(hypothesis.text, pipe.latency_report())
```
