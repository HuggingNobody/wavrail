"""CLI train subcommand: checkpoints round-trip through the CLI."""

from __future__ import annotations

import json

import pytest

from wavrail.cli.main import EXIT_OK, EXIT_RUNTIME_ERROR, main

pytestmark = pytest.mark.model


def test_train_saves_and_resumes_checkpoint(tmp_path, capsys) -> None:
    ckpt = tmp_path / "cli_ckpt.pt"
    code = main(
        [
            "train",
            "--epochs",
            "1",
            "--hidden-dim",
            "8",
            "--save-checkpoint",
            str(ckpt),
        ]
    )
    assert code == EXIT_OK
    assert ckpt.is_file()
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_train_examples"] == 3
    assert len(payload["epoch_losses"]) == 1

    code = main(["train", "--epochs", "2", "--hidden-dim", "8", "--resume", str(ckpt)])
    assert code == EXIT_OK
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["resumed_from_epoch"] == 1
    assert len(resumed["epoch_losses"]) == 2


def test_train_learning_rate_flag(tmp_path, capsys) -> None:
    code = main(["train", "--epochs", "1", "--hidden-dim", "8", "--learning-rate", "0.01"])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["learning_rates"] == [0.01]


def test_train_missing_checkpoint_reports_runtime_error(tmp_path, capsys) -> None:
    code = main(["train", "--epochs", "1", "--resume", str(tmp_path / "nope.pt")])
    assert code == EXIT_RUNTIME_ERROR
    assert "not found" in capsys.readouterr().err
