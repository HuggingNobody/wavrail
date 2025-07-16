"""Checkpoint save/load round-trips (requires the torch extra)."""

from __future__ import annotations

import pytest

from wavrail.errors import ModelError
from wavrail.model.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    Checkpoint,
    load_checkpoint,
    save_checkpoint,
)

pytestmark = pytest.mark.model


def _checkpoint(epoch: int = 2) -> Checkpoint:
    return Checkpoint(
        epoch=epoch,
        config={"n_epochs": 4, "seed": 0, "hidden_dim": 8},
        model_state={"gru": {}, "head": {}, "input_dim": 5, "hidden_dim": 8, "vocab_size": 4},
        epoch_losses=[1.2, 0.9],
        optimizer_state=None,
    )


def test_roundtrip_preserves_fields(tmp_path) -> None:
    path = tmp_path / "ckpt.pt"
    save_checkpoint(_checkpoint(), path)
    loaded = load_checkpoint(path)
    assert loaded.epoch == 2
    assert loaded.config == {"n_epochs": 4, "seed": 0, "hidden_dim": 8}
    assert loaded.epoch_losses == pytest.approx([1.2, 0.9])
    assert loaded.optimizer_state is None
    assert loaded.schema_version == CHECKPOINT_SCHEMA_VERSION


def test_load_missing_file_raises(tmp_path) -> None:
    with pytest.raises(ModelError, match="not found"):
        load_checkpoint(tmp_path / "nope.pt")


def test_load_corrupt_file_raises(tmp_path) -> None:
    path = tmp_path / "bad.pt"
    path.write_bytes(b"not a torch payload")
    with pytest.raises(ModelError, match="corrupt"):
        load_checkpoint(path)


def test_load_rejects_missing_keys(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "partial.pt"
    torch.save({"schema_version": CHECKPOINT_SCHEMA_VERSION, "epoch": 1}, path)
    with pytest.raises(ModelError, match="missing keys"):
        load_checkpoint(path)


def test_public_exports() -> None:
    import wavrail.model as model

    for name in (
        "Checkpoint",
        "save_checkpoint",
        "load_checkpoint",
        "CHECKPOINT_SCHEMA_VERSION",
    ):
        assert name in model.__all__
        assert hasattr(model, name)


def test_load_rejects_future_schema(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "future.pt"
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION + 1,
            "epoch": 1,
            "config": {},
            "model_state": {},
            "optimizer_state": None,
            "epoch_losses": [],
        },
        path,
    )
    with pytest.raises(ModelError, match="schema_version"):
        load_checkpoint(path)
