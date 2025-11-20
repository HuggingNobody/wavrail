"""JSON vocab persistence: roundtrips and schema enforcement."""

from __future__ import annotations

import json

import pytest

from wavrail.errors import VocabularyError
from wavrail.vocab import VOCAB_SCHEMA_VERSION, Vocab, load_vocab_json, save_vocab_json


def test_roundtrip_preserves_unicode_and_metadata(tmp_path):
    v = Vocab.from_labels(["<blank>", "你", "好", "a"], blank_index=0, name="zh")
    path = tmp_path / "vocab.json"
    save_vocab_json(v, path)
    assert load_vocab_json(path) == v


def test_file_carries_schema_version(tmp_path):
    path = tmp_path / "v.json"
    save_vocab_json(Vocab.from_labels(["<blank>", "x"]), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == VOCAB_SCHEMA_VERSION == 1


def test_unicode_stays_readable_in_file(tmp_path):
    path = tmp_path / "v.json"
    save_vocab_json(Vocab.from_labels(["<blank>", "你好"]), path)
    assert "你好" in path.read_text(encoding="utf-8")


def test_rejects_unknown_schema_version(tmp_path):
    path = tmp_path / "v.json"
    path.write_text(
        json.dumps({"schema_version": 99, "labels": ["<blank>", "a"]}), encoding="utf-8"
    )
    with pytest.raises(VocabularyError):
        load_vocab_json(path)


def test_rejects_invalid_json(tmp_path):
    path = tmp_path / "v.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(VocabularyError):
        load_vocab_json(path)


def test_rejects_non_object_json(tmp_path):
    path = tmp_path / "v.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(VocabularyError):
        load_vocab_json(path)
