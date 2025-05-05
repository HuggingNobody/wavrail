"""Corpus-level aggregation helpers.

A :class:`CorpusScore` accumulates per-utterance edit counts and
optionally slices by duration or noise level. The final report
exposes aggregate CER/WER plus optional per-slice breakdowns.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from wavrail.metrics.edit import EditOps, edit_alignment


@dataclass
class UtteranceScore:
    """Per-utterance edit counts and optional metadata."""

    name: str
    reference: str
    hypothesis: str
    ops: EditOps
    duration_seconds: float | None = None
    noise_level_db: float | None = None


@dataclass
class CorpusScore:
    """Aggregated corpus-level error statistics."""

    unit: str  # "character" or "word"
    utterances: list[UtteranceScore] = field(default_factory=list)

    def add(
        self,
        name: str,
        reference: str,
        hypothesis: str,
        *,
        duration_seconds: float | None = None,
        noise_level_db: float | None = None,
    ) -> UtteranceScore:
        if self.unit == "character":
            ref_units: list[str] = list(reference)
            hyp_units: list[str] = list(hypothesis)
        else:
            ref_units = reference.split()
            hyp_units = hypothesis.split()
        ops = edit_alignment(ref_units, hyp_units)
        score = UtteranceScore(
            name=name,
            reference=reference,
            hypothesis=hypothesis,
            ops=ops,
            duration_seconds=duration_seconds,
            noise_level_db=noise_level_db,
        )
        self.utterances.append(score)
        return score

    @property
    def total_errors(self) -> int:
        return sum(u.ops.errors for u in self.utterances)

    @property
    def total_length(self) -> int:
        return sum(u.ops.length for u in self.utterances)

    @property
    def error_rate(self) -> float:
        if self.total_length == 0:
            return 0.0
        return self.total_errors / self.total_length

    def slice_by_duration(self, threshold: float) -> dict[str, float]:
        """Split utterances by ``duration_seconds <= threshold``."""
        short: list[UtteranceScore] = []
        long: list[UtteranceScore] = []
        for u in self.utterances:
            bucket = short if (u.duration_seconds or 0.0) <= threshold else long
            bucket.append(u)
        return {"short": _bucket_rate(short), "long": _bucket_rate(long)}

    def slice_by_noise(self, threshold_db: float) -> dict[str, float]:
        """Split utterances by ``noise_level_db <= threshold_db``."""
        clean: list[UtteranceScore] = []
        noisy: list[UtteranceScore] = []
        for u in self.utterances:
            bucket = clean if (u.noise_level_db or 0.0) <= threshold_db else noisy
            bucket.append(u)
        return {"clean": _bucket_rate(clean), "noisy": _bucket_rate(noisy)}

    def per_utterance_rates(self) -> list[float]:
        rates: list[float] = []
        for u in self.utterances:
            if u.ops.length == 0:
                continue
            rates.append(u.ops.errors / u.ops.length)
        return rates

    def as_report(self, title: str = "score") -> dict[str, object]:
        return {
            "title": title,
            "unit": self.unit,
            "utterances": [
                {
                    "name": u.name,
                    "errors": u.ops.errors,
                    "length": u.ops.length,
                    "insertions": u.ops.insertions,
                    "deletions": u.ops.deletions,
                    "substitutions": u.ops.substitutions,
                    "duration_seconds": u.duration_seconds,
                    "noise_level_db": u.noise_level_db,
                }
                for u in self.utterances
            ],
            "total_errors": self.total_errors,
            "total_length": self.total_length,
            "error_rate": self.error_rate,
        }


def _bucket_rate(utterances: Iterable[UtteranceScore]) -> float:
    total_errors = 0
    total_length = 0
    for u in utterances:
        total_errors += u.ops.errors
        total_length += u.ops.length
    if total_length == 0:
        return 0.0
    return total_errors / total_length


def aggregate_score(unit: str = "character") -> CorpusScore:
    """Factory for an empty :class:`CorpusScore`."""
    if unit not in {"character", "word"}:
        raise ValueError("unit must be 'character' or 'word'")
    return CorpusScore(unit=unit)


__all__ = ["CorpusScore", "UtteranceScore", "aggregate_score"]
