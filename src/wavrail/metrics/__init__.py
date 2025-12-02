"""Text normalization and error-rate metrics.

The metrics in this subpackage all share a single
:class:`TextNormalizer` so that evaluation and training see the
same canonical form of an utterance. The two error-rate
functions are :func:`character_error_rate` (used for Chinese and
other character-based languages) and :func:`word_error_rate`
(used for English and other space-delimited languages).

Edit-alignment output is available via
:func:`edit_alignment` and a corpus-level summary is produced by
:class:`CorpusScore`.
"""

from wavrail.metrics.cer import character_error_rate
from wavrail.metrics.confusion import ConfusionReport, confusion_report
from wavrail.metrics.corpus import CorpusScore, UtteranceScore, aggregate_score
from wavrail.metrics.edit import EditOps, edit_alignment
from wavrail.metrics.normalization import (
    TextNormalizer,
    fullwidth_to_halfwidth,
    halfwidth_to_fullwidth,
    normalize_digits,
)
from wavrail.metrics.wer import word_error_rate

__all__ = [
    "ConfusionReport",
    "CorpusScore",
    "EditOps",
    "TextNormalizer",
    "UtteranceScore",
    "aggregate_score",
    "character_error_rate",
    "confusion_report",
    "edit_alignment",
    "fullwidth_to_halfwidth",
    "halfwidth_to_fullwidth",
    "normalize_digits",
    "word_error_rate",
]
