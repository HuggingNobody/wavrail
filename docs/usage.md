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
