"""Token-level confusion accounting on top of edit alignments.

Error rates answer "how many"; confusion reports answer "which".
:class:`ConfusionReport` counts matches, substitutions (as
reference -> hypothesis pairs), insertions, and deletions per token,
so systematic mix-ups — acoustically similar characters, homophones —
stand out instead of hiding inside an aggregate rate. The input is a
:class:`wavrail.metrics.EditOps`, so both character and word units
are supported.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from wavrail.metrics.edit import EditOps, OpType


@dataclass(frozen=True)
class ConfusionReport:
    """Per-token counts of each edit category.

    Attributes
    ----------
    matches:
        Counter of correctly recognised reference tokens.
    substitutions:
        Counter keyed by ``(reference, hypothesis)`` pairs.
    insertions:
        Counter of extra hypothesis tokens.
    deletions:
        Counter of missing reference tokens.
    """

    matches: Counter[str]
    substitutions: Counter[tuple[str, str]]
    insertions: Counter[str]
    deletions: Counter[str]

    @classmethod
    def from_ops(cls, ops: EditOps) -> ConfusionReport:
        """Tally every edit operation of ``ops`` into the counters."""
        matches: Counter[str] = Counter()
        substitutions: Counter[tuple[str, str]] = Counter()
        insertions: Counter[str] = Counter()
        deletions: Counter[str] = Counter()
        for edit in ops.ops:
            if edit.op == OpType.MATCH:
                matches[str(edit.reference)] += 1
            elif edit.op == OpType.SUBSTITUTION:
                substitutions[(str(edit.reference), str(edit.hypothesis))] += 1
            elif edit.op == OpType.INSERTION:
                insertions[str(edit.hypothesis)] += 1
            elif edit.op == OpType.DELETION:
                deletions[str(edit.reference)] += 1
        return cls(
            matches=matches,
            substitutions=substitutions,
            insertions=insertions,
            deletions=deletions,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready payload with deterministic key order."""
        return {
            "matches": dict(sorted(self.matches.items())),
            "substitutions": [
                {"reference": ref, "hypothesis": hyp, "count": count}
                for (ref, hyp), count in sorted(self.substitutions.items())
            ],
            "insertions": dict(sorted(self.insertions.items())),
            "deletions": dict(sorted(self.deletions.items())),
        }


def confusion_report(ops: EditOps) -> ConfusionReport:
    """Convenience wrapper for :meth:`ConfusionReport.from_ops`."""
    return ConfusionReport.from_ops(ops)


__all__ = ["ConfusionReport", "confusion_report"]
