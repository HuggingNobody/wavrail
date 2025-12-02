"""ConfusionReport category counting and dict contract."""

from __future__ import annotations

import json

from wavrail.metrics import confusion_report, edit_alignment


def test_substitution_pair_counted():
    ops = edit_alignment(list("abcd"), list("axcd"))
    report = confusion_report(ops)
    assert report.substitutions[("b", "x")] == 1
    assert report.matches["a"] == 1
    assert report.matches["c"] == 1
    assert report.matches["d"] == 1


def test_insertion_and_deletion_tokens():
    inserted = confusion_report(edit_alignment(list("ab"), list("acb")))
    assert inserted.insertions["c"] == 1
    deleted = confusion_report(edit_alignment(list("abc"), list("ac")))
    assert deleted.deletions["b"] == 1


def test_word_units_supported():
    ops = edit_alignment(["the", "cat", "sat"], ["the", "dog", "sat"])
    report = confusion_report(ops)
    assert report.substitutions[("cat", "dog")] == 1
    assert report.matches["the"] == 1


def test_to_dict_is_json_ready_and_deterministic():
    ops = edit_alignment(list("ab"), list("xb"))
    payload = confusion_report(ops).to_dict()
    assert payload["substitutions"] == [{"reference": "a", "hypothesis": "x", "count": 1}]
    assert payload["matches"] == {"b": 1}
    assert json.loads(json.dumps(payload)) == payload


def test_repeated_confusion_accumulates():
    ops = edit_alignment(list("aaab"), list("xxxb"))
    report = confusion_report(ops)
    assert report.substitutions[("a", "x")] == 3
