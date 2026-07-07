"""Schema versioning, JSON serialisation, and report export.

The wavrail public outputs (hypotheses, alignments, score reports)
all carry a ``schema_version`` field and a JSON-friendly
representation so they can be persisted and validated without
depending on the Python runtime. Each public dataclass has a
``to_dict``/``from_dict`` pair and a ``SCHEMA_VERSION`` constant
that downstream consumers can use to gate compatibility.

The export helpers add JSONL interchange (``read_jsonl`` /
``write_jsonl``) and offline Markdown rendering
(``hypotheses_to_markdown``, ``score_report_to_markdown``,
``alignments_to_markdown``) for the same payloads.
"""

from wavrail.report.export import (
    alignment_record,
    alignments_to_markdown,
    hypotheses_to_markdown,
    hypothesis_record,
    markdown_table,
    read_jsonl,
    score_report_to_markdown,
    to_jsonl,
    write_jsonl,
)
from wavrail.report.schema import (
    SCHEMA_VERSION,
    SchemaError,
    align_from_dict,
    align_to_dict,
    alignment_summary_to_dict,
    hypothesis_from_dict,
    hypothesis_to_dict,
    score_report_from_json,
    score_report_to_dict,
    score_report_to_json,
    validate_score_report,
)

__all__ = [
    "SCHEMA_VERSION",
    "SchemaError",
    "align_from_dict",
    "align_to_dict",
    "alignment_record",
    "alignment_summary_to_dict",
    "alignments_to_markdown",
    "hypotheses_to_markdown",
    "hypothesis_from_dict",
    "hypothesis_record",
    "hypothesis_to_dict",
    "markdown_table",
    "read_jsonl",
    "score_report_from_json",
    "score_report_to_dict",
    "score_report_to_json",
    "score_report_to_markdown",
    "to_jsonl",
    "validate_score_report",
    "write_jsonl",
]
