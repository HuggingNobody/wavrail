"""Train CLI warmup and early-stop flags (torch extra)."""

from __future__ import annotations

import json

import pytest

from wavrail.cli.main import EXIT_OK, main

pytestmark = pytest.mark.model


def test_warmup_flag_ramps_learning_rate(capsys) -> None:
    code = main(
        [
            "train",
            "--epochs",
            "3",
            "--hidden-dim",
            "8",
            "--learning-rate",
            "0.01",
            "--warmup-epochs",
            "2",
        ]
    )
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["learning_rates"][0] == pytest.approx(0.005)
    assert payload["learning_rates"][1] == pytest.approx(0.01)


def test_early_stop_flags_stop_plateaued_run(capsys) -> None:
    code = main(
        [
            "train",
            "--epochs",
            "20",
            "--hidden-dim",
            "4",
            "--learning-rate",
            "1e-4",
            "--early-stop-patience",
            "2",
            "--early-stop-min-delta",
            "0.5",
        ]
    )
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["stopped_early"] is True
    assert len(payload["epoch_losses"]) == 3
