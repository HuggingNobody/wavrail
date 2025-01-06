"""Exception hierarchy for the wavrail package.

Defining a stable hierarchy makes it easy for downstream code to catch
wavrail-specific failures without masking unrelated errors.
"""

from __future__ import annotations


class WavrailError(Exception):
    """Base class for all wavrail-specific errors."""


class AudioError(WavrailError):
    """Raised for audio I/O, format, and conversion problems."""


class FormatError(AudioError):
    """Raised when a WAV header advertises an unsupported encoding."""


class FeatureError(WavrailError):
    """Raised for feature-extraction configuration or shape problems."""


class VocabularyError(WavrailError):
    """Raised for vocabulary construction and token<->id conversion failures."""


class DecodingError(WavrailError):
    """Raised for decoder configuration and execution failures."""


class AlignmentError(WavrailError):
    """Raised for forced-alignment failures (impossible transcripts, empty lattice)."""


class MetricError(WavrailError):
    """Raised for metric configuration and computation problems."""


class PipelineError(WavrailError):
    """Raised for streaming pipeline state and lifecycle failures."""


class ModelError(WavrailError):
    """Raised for acoustic model construction and training failures."""


class SchemaError(WavrailError):
    """Raised for report-schema validation or version mismatches."""
