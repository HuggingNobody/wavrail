"""Schema versioning and JSON serialisation helpers.

The wavrail public outputs (hypotheses, alignments, score reports)
all carry a ``schema_version`` field and a JSON-friendly
representation so they can be persisted and validated without
depending on the Python runtime. Each public dataclass has a
``to_dict``/``from_dict`` pair and a ``SCHEMA_VERSION`` constant
that downstream consumers can use to gate compatibility.
"""

from wavrail.report.schema import (
    SCHEMA_VERSION,
    SchemaError,
    align_from_dict,
    align_to_dict,
    hypothesis_from_dict,
    hypothesis_to_dict,
    score_report_from_json,
    score_report_to_json,
)

__all__ = [
    "SCHEMA_VERSION",
    "SchemaError",
    "align_from_dict",
    "align_to_dict",
    "hypothesis_from_dict",
    "hypothesis_to_dict",
    "score_report_from_json",
    "score_report_to_json",
]
