"""Inference helpers bridging trained models and CTC decoders.

:func:`model_posteriors` runs an :class:`AcousticModel` (or any
module with the same forward contract) over log-mel features and
returns a probability matrix with the decoder's ``(num_frames,
vocab_size)`` shape. :func:`decode_example` wires that matrix into
greedy or prefix-beam decoding so a checkpoint can be evaluated on a
synthetic example in one call. Both require the optional torch extra.
"""

from __future__ import annotations

import numpy as np

from wavrail.ctc import Hypothesis, PrefixBeamSearch, greedy_decode
from wavrail.errors import ModelError
from wavrail.model.synthetic import SyntheticExample
from wavrail.vocab import Vocab

_LOG_FLOOR = 1e-20


def _require_torch():
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ModelError(
            "the optional 'torch' extra is required for model inference; "
            "install with `pip install wavrail[torch]`"
        ) from exc
    return torch


def model_posteriors(model: object, features: np.ndarray) -> np.ndarray:
    """Return ``(num_frames, vocab_size)`` class probabilities."""
    torch = _require_torch()
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ModelError("features must be a non-empty (num_frames, num_mels) array")
    model.eval()  # type: ignore[attr-defined]
    with torch.no_grad():
        logits = model(torch.tensor(arr[None, ...], dtype=torch.float32))  # type: ignore[operator]
        probs = torch.softmax(logits, dim=-1)[0]
    return np.asarray(probs.detach().numpy(), dtype=np.float32)


def decode_example(
    model: object,
    example: SyntheticExample,
    vocab: Vocab,
    *,
    beam: int = 1,
) -> Hypothesis:
    """Decode one synthetic example end to end.

    ``beam <= 1`` selects greedy decoding; larger values run the
    prefix beam search on floored log-probabilities. A model whose
    output width disagrees with the vocab is rejected instead of
    silently decoding nonsense.
    """
    posteriors = model_posteriors(model, example.features)
    if posteriors.shape[1] != vocab.size:
        raise ModelError(f"model emits {posteriors.shape[1]} classes but vocab has {vocab.size}")
    if beam <= 1:
        return greedy_decode(posteriors, vocab)
    log_probs = np.log(np.maximum(posteriors, _LOG_FLOOR))
    return PrefixBeamSearch(vocab=vocab, beam=beam).decode(log_probs)


__all__ = ["decode_example", "model_posteriors"]
