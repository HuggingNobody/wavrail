"""Checkpoint persistence for the synthetic trainer.

A checkpoint is a ``torch.save`` payload holding the model weights,
optimiser state, the number of completed epochs, the training
config, and the loss history. Checkpoints exist so the demo
trainer can be paused and resumed deterministically; the schema is
versioned and validated on load so stale or truncated files fail
loudly instead of half-restoring.

Checkpoint files are trusted local artefacts: loading uses
``weights_only=False`` and must only ever point at files this
library (or the user) wrote.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wavrail.errors import ModelError

CHECKPOINT_SCHEMA_VERSION = 1

_REQUIRED_KEYS = frozenset(
    {"schema_version", "epoch", "config", "model_state", "optimizer_state", "epoch_losses"}
)


def _require_torch():
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ModelError(
            "the optional 'torch' extra is required for checkpoints; "
            "install with `pip install wavrail[torch]`"
        ) from exc
    return torch


@dataclass
class Checkpoint:
    """In-memory representation of a training checkpoint."""

    epoch: int
    config: dict[str, Any]
    model_state: dict[str, Any]
    epoch_losses: list[float] = field(default_factory=list)
    optimizer_state: dict[str, Any] | None = None
    schema_version: int = CHECKPOINT_SCHEMA_VERSION


def save_checkpoint(checkpoint: Checkpoint, path: str | Path) -> None:
    """Write ``checkpoint`` to ``path``, creating parent directories."""
    torch = _require_torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": int(checkpoint.schema_version),
        "epoch": int(checkpoint.epoch),
        "config": dict(checkpoint.config),
        "model_state": checkpoint.model_state,
        "optimizer_state": checkpoint.optimizer_state,
        "epoch_losses": [float(v) for v in checkpoint.epoch_losses],
    }
    torch.save(payload, target)


def load_checkpoint(path: str | Path) -> Checkpoint:
    """Load and validate the checkpoint stored at ``path``."""
    torch = _require_torch()
    target = Path(path)
    if not target.is_file():
        raise ModelError(f"checkpoint not found: {target}")
    try:
        payload = torch.load(target, weights_only=False)
    except Exception as exc:
        raise ModelError(f"checkpoint is corrupt: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelError("checkpoint payload must be a dict")
    missing = _REQUIRED_KEYS - set(payload)
    if missing:
        raise ModelError(f"checkpoint is missing keys: {sorted(missing)}")
    version = payload["schema_version"]
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise ModelError(
            f"incompatible checkpoint schema_version={version!r}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION!r}"
        )
    return Checkpoint(
        epoch=int(payload["epoch"]),
        config=dict(payload["config"]),
        model_state=payload["model_state"],
        epoch_losses=[float(v) for v in payload["epoch_losses"]],
        optimizer_state=payload["optimizer_state"],
        schema_version=int(version),
    )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "Checkpoint",
    "load_checkpoint",
    "save_checkpoint",
]
