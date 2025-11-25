"""Vocabularies for label<->id conversion.

A :class:`Vocab` is constructed from a list of labels (the first
entry is treated as the CTC blank by convention; callers can
override ``blank_index`` if they wish to use a different
convention). All strings are stored as ``str`` (Unicode), which
makes the vocabulary safe for non-ASCII scripts such as Chinese.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from wavrail.errors import VocabularyError


@dataclass(frozen=True)
class Vocab:
    labels: tuple[str, ...]
    blank_index: int = 0
    name: str = field(default="vocab")

    def __post_init__(self) -> None:
        if not self.labels:
            raise VocabularyError("Vocab must have at least one label")
        if not 0 <= self.blank_index < len(self.labels):
            raise VocabularyError("blank_index out of range")
        if len(set(self.labels)) != len(self.labels):
            raise VocabularyError("Vocab labels must be unique")

    @classmethod
    def from_labels(
        cls,
        labels: Iterable[str],
        blank_index: int = 0,
        name: str = "vocab",
    ) -> Vocab:
        return cls(labels=tuple(labels), blank_index=blank_index, name=name)

    @classmethod
    def from_text(
        cls,
        text: str,
        blank_token: str = "<blank>",
        name: str = "vocab",
    ) -> Vocab:
        """Build a character-level vocab from ``text`` in first-seen order.

        The blank token always occupies index 0. Remaining labels are
        single characters appended in order of first appearance, which
        keeps the layout deterministic for a given corpus. Whitespace
        and punctuation count as ordinary characters; callers who want
        them excluded should normalise the text first. Text that adds
        no character beyond the blank token is rejected because the
        resulting blank-only vocab could not decode anything.
        """
        labels = [blank_token]
        seen = {blank_token}
        for char in text:
            if char not in seen:
                seen.add(char)
                labels.append(char)
        if len(labels) == 1:
            raise VocabularyError("from_text found no characters beyond the blank token")
        return cls(labels=tuple(labels), blank_index=0, name=name)

    @property
    def size(self) -> int:
        return len(self.labels)

    def token_to_id(self, token: str) -> int:
        try:
            return self.labels.index(token)
        except ValueError as exc:
            raise VocabularyError(f"unknown token: {token!r}") from exc

    def id_to_token(self, index: int) -> str:
        if not 0 <= index < len(self.labels):
            raise VocabularyError(f"index out of range: {index}")
        return self.labels[index]

    def is_blank(self, index: int) -> bool:
        return index == self.blank_index

    def to_dict(self) -> dict[str, object]:
        return {
            "labels": list(self.labels),
            "blank_index": self.blank_index,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Vocab:
        labels = payload.get("labels") or []
        return cls(
            labels=tuple(str(label) for label in labels),
            blank_index=int(payload.get("blank_index", 0)),
            name=str(payload.get("name", "vocab")),
        )

    def __len__(self) -> int:
        return self.size


__all__ = ["Vocab"]
