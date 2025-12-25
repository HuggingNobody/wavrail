"""top_substitutions ranking and validation."""

from __future__ import annotations

import pytest

from wavrail.errors import MetricError
from wavrail.metrics import confusion_report, edit_alignment, top_substitutions


def test_orders_by_count():
    ops = edit_alignment(list("aaabcd"), list("xxxycd"))
    report = confusion_report(ops)
    top = top_substitutions(report, k=2)
    assert top[0] == (("a", "x"), 3)
    assert top[1] == (("b", "y"), 1)


def test_k_larger_than_distinct_pairs():
    ops = edit_alignment(list("ab"), list("xb"))
    assert top_substitutions(confusion_report(ops), k=99) == [(("a", "x"), 1)]


def test_empty_when_no_substitutions():
    ops = edit_alignment(list("abc"), list("abc"))
    assert top_substitutions(confusion_report(ops)) == []


def test_rejects_non_positive_k():
    report = confusion_report(edit_alignment(list("a"), list("b")))
    with pytest.raises(MetricError):
        top_substitutions(report, k=0)
