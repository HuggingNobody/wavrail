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

config = ft.MelConfig(sample_rate=16000, num_mels=80, fft_size=512)
log_mel = ft.log_mel_spectrogram(samples, sample_rate=16000, config=config)
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

report = metrics.score(reference="hello world",
                       hypothesis="hello wurld")
print(report.wer)
```

## Streaming

```python
import wavrail.streaming as streaming

pipe = streaming.Pipeline(...)  # configured elsewhere
for chunk in audio.chunks(samples, chunk_size=1600):
    pipe.feed(chunk)
    while pipe.has_emission():
        print(pipe.next_emission())
```
