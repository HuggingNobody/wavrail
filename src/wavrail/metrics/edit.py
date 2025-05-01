"""Levenshtein-style edit alignment.

Given two sequences ``reference`` and ``hypothesis`` we compute
the minimum-edit-distance alignment and return it as a list of
:class:`EditOps`. The implementation is the textbook dynamic
program over ``len(reference)+1`` by ``len(hypothesis)+1`` and
runs in O(R*H) time and space.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum


class OpType(StrEnum):
    """Edit operation kinds."""

    MATCH = "match"
    SUBSTITUTION = "substitution"
    INSERTION = "insertion"
    DELETION = "deletion"


@dataclass(frozen=True)
class EditOp:
    """A single edit operation in an alignment."""

    op: OpType
    reference: object
    hypothesis: object


@dataclass(frozen=True)
class EditOps:
    """Container for an edit-alignment result.

    Attributes
    ----------
    ops:
        Sequence of :class:`EditOp` covering the entire alignment.
    insertions:
        Number of insertion operations (extra hypothesis tokens).
    deletions:
        Number of deletion operations (missing hypothesis tokens).
    substitutions:
        Number of substitution operations (wrong hypothesis tokens).
    length:
        Reference length used as the denominator in error rates.
    """

    ops: tuple[EditOp, ...]
    insertions: int
    deletions: int
    substitutions: int
    length: int

    @property
    def errors(self) -> int:
        return self.insertions + self.deletions + self.substitutions


def edit_alignment(
    reference: Sequence[object],
    hypothesis: Sequence[object],
) -> EditOps:
    """Compute the Levenshtein edit alignment of ``reference`` and ``hypothesis``."""
    rows = len(reference)
    cols = len(hypothesis)
    cost = [[0] * (cols + 1) for _ in range(rows + 1)]
    back: list[list[tuple[int, int] | None]] = [[None] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        cost[i][0] = i
        if i > 0:
            back[i][0] = (i - 1, 0)
    for j in range(cols + 1):
        cost[0][j] = j
        if j > 0:
            back[0][j] = (0, j - 1)
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            ref_tok = reference[i - 1]
            hyp_tok = hypothesis[j - 1]
            diag = cost[i - 1][j - 1] + (0 if ref_tok == hyp_tok else 1)
            up = cost[i - 1][j] + 1
            left = cost[i][j - 1] + 1
            best = min(diag, up, left)
            cost[i][j] = best
            if best == diag:
                back[i][j] = (i - 1, j - 1)
            elif best == up:
                back[i][j] = (i - 1, j)
            else:
                back[i][j] = (i, j - 1)
    ops: list[EditOp] = []
    insertions = deletions = substitutions = 0
    i, j = rows, cols
    while i > 0 or j > 0:
        prev = back[i][j]
        if prev is None:
            break
        pi, pj = prev
        if pi == i - 1 and pj == j - 1:
            ref_tok = reference[i - 1]
            hyp_tok = hypothesis[j - 1]
            if ref_tok == hyp_tok:
                ops.append(EditOp(OpType.MATCH, ref_tok, hyp_tok))
            else:
                ops.append(EditOp(OpType.SUBSTITUTION, ref_tok, hyp_tok))
                substitutions += 1
        elif pi == i - 1 and pj == j:
            ops.append(EditOp(OpType.DELETION, reference[i - 1], None))
            deletions += 1
        else:
            ops.append(EditOp(OpType.INSERTION, None, hypothesis[j - 1]))
            insertions += 1
        i, j = pi, pj
    ops.reverse()
    return EditOps(
        ops=tuple(ops),
        insertions=insertions,
        deletions=deletions,
        substitutions=substitutions,
        length=rows,
    )


__all__ = ["EditOp", "EditOps", "OpType", "edit_alignment"]
