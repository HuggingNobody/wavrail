"""Compact acoustic model with the optional ``torch`` extra.

The :class:`AcousticModel` is a small encoder (GRU by default
with an optional ``conformer-lite`` configuration) followed by a
linear CTC head over log-mel features. The module depends on
PyTorch, which is an *optional* extra of wavrail: importing this
subpackage without ``torch`` installed raises a clear
:class:`ModelError`.

The training entry point :func:`train_synthetic` demonstrates a
deterministic, small-scale training loop on a fully synthetic
formant dataset. The synthetic dataset is explicitly labelled as
*not real speech*; the only purpose is to exercise the training
loop end-to-end.
"""

from wavrail.model.synthetic import (
    SyntheticExample,
    build_synthetic_dataset,
)
from wavrail.model.train import (
    TrainingConfig,
    TrainingReport,
    train_synthetic,
)

__all__ = [
    "SyntheticExample",
    "TrainingConfig",
    "TrainingReport",
    "build_synthetic_dataset",
    "train_synthetic",
]
