"""Seeded property: confusion totals reconcile with EditOps counters."""

from __future__ import annotations

import pytest

from wavrail.metrics import confusion_report, edit_alignment


@pytest.mark.parametrize("seed", range(6))
def test_totals_reconcile(seed: int) -> None:
    import numpy as np

    rng = np.random.default_rng(seed)
    alphabet = list("abcd")
    ref = [str(rng.choice(alphabet)) for _ in range(int(rng.integers(1, 12)))]
    hyp = [str(rng.choice(alphabet)) for _ in range(int(rng.integers(0, 12)))]
    ops = edit_alignment(ref, hyp)
    report = confusion_report(ops)

    assert (
        sum(report.matches.values())
        + sum(report.substitutions.values())
        + sum(report.deletions.values())
        == ops.length
    )
    assert sum(report.substitutions.values()) == ops.substitutions
    assert sum(report.insertions.values()) == ops.insertions
    assert sum(report.deletions.values()) == ops.deletions
    assert sum(report.matches.values()) == ops.length - ops.substitutions - ops.deletions
