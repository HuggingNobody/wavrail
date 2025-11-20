"""File persistence for vocabularies.

Two formats are supported:

- **JSON** (``*.json``): the dictionary form of :class:`Vocab` plus an
  explicit ``schema_version`` field. Loaders reject unknown versions
  so a corpus-side vocab file can never silently drift from the code.
- **tokens** (``*.txt``): one label per line, where the label index is
  its line number — the layout used by most CTC training recipes. The
  line format cannot express ``blank_index`` or ``name``, so loaders
  take them as arguments. Labels containing newlines are rejected on
  save; use JSON for those.

Both formats are UTF-8 and written with ``ensure_ascii=False`` so
Chinese labels stay readable in the file.
"""

from __future__ import annotations

import json
from pathlib import Path

from wavrail.errors import VocabularyError
from wavrail.vocab.vocab import Vocab

VOCAB_SCHEMA_VERSION = 1


def save_vocab_json(vocab: Vocab, path: str | Path) -> None:
    """Write ``vocab`` to ``path`` as versioned JSON."""
    payload = {"schema_version": VOCAB_SCHEMA_VERSION, **vocab.to_dict()}
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_vocab_json(path: str | Path) -> Vocab:
    """Read a versioned JSON vocab; reject unknown schema versions."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VocabularyError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise VocabularyError("vocab JSON must be a single object")
    version = payload.get("schema_version")
    if version != VOCAB_SCHEMA_VERSION:
        raise VocabularyError(
            f"incompatible schema_version={version!r}; expected {VOCAB_SCHEMA_VERSION!r}"
        )
    return Vocab.from_dict(payload)


def save_vocab_tokens(vocab: Vocab, path: str | Path) -> None:
    """Write ``vocab`` as one label per line (index = line number)."""
    for label in vocab.labels:
        if "\n" in label or "\r" in label:
            raise VocabularyError(
                f"label {label!r} contains a newline; use the JSON format instead"
            )
    Path(path).write_text("".join(f"{label}\n" for label in vocab.labels), encoding="utf-8")


def load_vocab_tokens(
    path: str | Path,
    blank_index: int = 0,
    name: str = "vocab",
) -> Vocab:
    """Read a line-format tokens file into a :class:`Vocab`."""
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        raise VocabularyError(f"{path} contains no tokens")
    return Vocab(labels=tuple(lines), blank_index=blank_index, name=name)


__all__ = [
    "VOCAB_SCHEMA_VERSION",
    "load_vocab_json",
    "load_vocab_tokens",
    "save_vocab_json",
    "save_vocab_tokens",
]
