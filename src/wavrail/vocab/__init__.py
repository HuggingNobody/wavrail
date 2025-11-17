"""Token<->id vocabularies used by the CTC decoders.

The :class:`wavrail.vocab.Vocab` class wraps a frozen list of
unicode-safe labels and a blank index. It supports deterministic
serialisation to and from plain Python dictionaries so that
trained checkpoints can be reloaded without rebuilding the
mapping from the underlying labels file.
"""

from wavrail.vocab.persist import VOCAB_SCHEMA_VERSION, load_vocab_json, save_vocab_json
from wavrail.vocab.vocab import Vocab

__all__ = ["VOCAB_SCHEMA_VERSION", "Vocab", "load_vocab_json", "save_vocab_json"]
