"""Frame-synchronous (transducer-style) emission decoder.

The decoder walks the posterior matrix frame by frame and emits
tokens once enough evidence has accumulated. Compared with a
plain greedy decoder the emission is delayed by a configurable
number of frames so that a downstream consumer can see stable
hypotheses as the audio streams.

This subpackage deliberately avoids building a full label
lattice: only the single best hypothesis is tracked, with an
optional *emission delay* (in frames) that controls how long a
candidate must remain the argmax before it is committed.
"""

from wavrail.emit.streaming import (
    StreamingDecoder,
    StreamingDecoderState,
    decode_streaming,
    deserialize_state,
    iter_chunks,
    serialize_state,
)

__all__ = [
    "StreamingDecoder",
    "StreamingDecoderState",
    "decode_streaming",
    "deserialize_state",
    "iter_chunks",
    "serialize_state",
]
