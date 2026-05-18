"""Compact acoustic model with the optional ``torch`` extra.

The :class:`AcousticModel` is a small two-layer GRU encoder
followed by a linear CTC head over log-mel features. PyTorch is
an *optional* extra of wavrail: importing
:mod:`wavrail.model.network` or :mod:`wavrail.model.checkpoint`
helpers without ``torch`` installed raises a clear
:class:`ModelError`, while the config and report dataclasses stay
importable.

Checkpoints (:func:`save_checkpoint` / :func:`load_checkpoint`)
make a training run resumable; the resume path is covered by
equivalence tests against uninterrupted runs.

The training entry point :func:`train_synthetic` demonstrates a
deterministic, small-scale training loop on a fully synthetic
formant dataset. The synthetic dataset is explicitly labelled as
*not real speech*; the only purpose is to exercise the training
loop end-to-end.
"""

from wavrail.model.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    Checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from wavrail.model.earlystop import EarlyStopper
from wavrail.model.infer import decode_example, model_posteriors
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
    "CHECKPOINT_SCHEMA_VERSION",
    "Checkpoint",
    "EarlyStopper",
    "SyntheticExample",
    "TrainingConfig",
    "TrainingReport",
    "build_synthetic_dataset",
    "decode_example",
    "load_checkpoint",
    "model_posteriors",
    "save_checkpoint",
    "train_synthetic",
]
