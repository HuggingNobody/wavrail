"""Line-format tokens persistence."""

from __future__ import annotations

import pytest

from wavrail.errors import VocabularyError
from wavrail.vocab import Vocab, load_vocab_tokens, save_vocab_tokens


def test_roundtrip_line_format(tmp_path):
    v = Vocab.from_labels(["<blank>", "你", "好", " "], name="t")
    path = tmp_path / "tokens.txt"
    save_vocab_tokens(v, path)
    loaded = load_vocab_tokens(path)
    assert loaded.labels == v.labels
    assert loaded.blank_index == 0


def test_index_matches_line_number(tmp_path):
    path = tmp_path / "tokens.txt"
    path.write_text("<blank>\na\nb\n", encoding="utf-8")
    v = load_vocab_tokens(path)
    assert v.token_to_id("b") == 2
    assert v.id_to_token(1) == "a"


def test_blank_index_and_name_are_loader_arguments(tmp_path):
    path = tmp_path / "tokens.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    v = load_vocab_tokens(path, blank_index=2, name="custom")
    assert v.blank_index == 2
    assert v.name == "custom"
    assert v.is_blank(2)


def test_rejects_labels_with_newlines(tmp_path):
    v = Vocab.from_labels(["<blank>", "a\nb"])
    with pytest.raises(VocabularyError):
        save_vocab_tokens(v, tmp_path / "t.txt")


def test_rejects_empty_file(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("", encoding="utf-8")
    with pytest.raises(VocabularyError):
        load_vocab_tokens(path)
