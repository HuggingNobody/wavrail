"""Streaming feature-and-decode pipeline.

A :class:`StreamingPipeline` ties together a frame blocker, a
feature extractor, and an emission decoder behind a single
``feed_samples`` API. It keeps a ring buffer of audio samples so
that successive chunks can be processed with bounded lookahead
and the pipeline state can be paused/resumed across calls.

The default ``measure_rtf`` helper exposes real-time factor
calculation by timing each ``feed_samples`` call and dividing
by the audio duration that was processed.
"""

from wavrail.stream.buffer import RingBuffer
from wavrail.stream.framer import (
    FRAMER_STATE_VERSION,
    StreamingFramer,
    deserialize_framer_state,
    serialize_framer_state,
)
from wavrail.stream.pipeline import (
    LatencyReport,
    StreamingPipeline,
    StreamingPipelineState,
    deserialize_pipeline_state,
    measure_rtf,
    serialize_pipeline_state,
)
from wavrail.stream.spectrogram import StreamingSpectrogram

__all__ = [
    "FRAMER_STATE_VERSION",
    "LatencyReport",
    "RingBuffer",
    "StreamingFramer",
    "StreamingPipeline",
    "StreamingPipelineState",
    "StreamingSpectrogram",
    "deserialize_framer_state",
    "deserialize_pipeline_state",
    "measure_rtf",
    "serialize_framer_state",
    "serialize_pipeline_state",
]
