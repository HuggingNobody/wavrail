"""Acoustic model network definition.

A compact 2-layer GRU encoder followed by a linear CTC head.
Importing this module without the optional ``torch`` extra raises
a :class:`wavrail.errors.ModelError`.
"""

from __future__ import annotations

from wavrail.errors import ModelError


def _require_torch():
    try:
        import torch  # type: ignore[import-not-found]
        import torch.nn as nn  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ModelError(
            "the optional 'torch' extra is required for the acoustic model; "
            "install with `pip install wavrail[torch]`"
        ) from exc
    return torch, nn


_ModuleBase = _require_torch()[1].Module


class AcousticModel(_ModuleBase):  # type: ignore[misc, valid-type]
    """Compact GRU encoder + CTC head.

    The class is constructed only when torch is available. The
    forward method returns a tensor of shape
    ``(batch, time, vocab_size)`` containing raw logits.
    """

    def __init__(self, *, input_dim: int, hidden_dim: int, vocab_size: int) -> None:
        _torch, nn = _require_torch()
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):  # type: ignore[no-untyped-def]
        _torch, _ = _require_torch()
        outputs, _h = self.gru(x)
        return self.head(outputs)

    def state_dict(self):  # type: ignore[no-untyped-def]
        _torch, _ = _require_torch()
        return {
            "gru": self.gru.state_dict(),
            "head": self.head.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "vocab_size": self.vocab_size,
        }

    def load_state_dict(self, payload):  # type: ignore[no-untyped-def]
        self.gru.load_state_dict(payload["gru"])
        self.head.load_state_dict(payload["head"])


def build_targets(examples):  # type: ignore[no-untyped-def]
    """Concatenate token sequences into a single (targets, lengths) tensor."""
    torch, _ = _require_torch()
    lengths = [len(example.tokens) for example in examples]
    targets = torch.tensor(
        [tok for example in examples for tok in example.tokens], dtype=torch.long
    )
    return targets, torch.tensor(lengths, dtype=torch.long)


__all__ = ["AcousticModel", "build_targets"]
