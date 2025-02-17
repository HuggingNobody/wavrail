"""Signal processing features for speech recognition.

This subpackage builds on the audio primitives exposed by
:mod:`wavrail.audio`. It provides frame blocking, window functions,
pre-emphasis, the power spectrum, mel filterbank construction,
log-mel features, streaming cepstral mean and variance
normalisation (CMVN), and delta derivatives.

Public entry points are added incrementally as the corresponding
modules land; see ``__all__`` for the current surface.
"""

__all__ = []
