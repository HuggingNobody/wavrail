"""Top-level package for the wavrail streaming speech recognition toolkit.

Public entry points and version metadata live here. Subpackages expose the
functional API for audio I/O, feature extraction, vocabulary handling,
CTC decoding, metrics, streaming pipelines, and the optional acoustic
model implemented in pure NumPy plus the optional torch extra.
"""

from wavrail._version import __version__

__all__ = ["__version__"]
