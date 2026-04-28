"""Patience-based early stopping for the training loop.

:class:`EarlyStopper` is pure Python with no torch dependency, so the
stopping policy is unit-testable without the model extra installed.
The metric is lower-is-better (training loss); an improvement must
beat the best value by at least ``min_delta`` to reset the waiting
counter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from wavrail.errors import ModelError


@dataclass
class EarlyStopper:
    """Decide when more training cannot help.

    ``step(value)`` records one metric observation and returns
    ``True`` once the metric has failed to improve by at least
    ``min_delta`` for ``patience`` consecutive steps.
    ``patience = 0`` disables stopping entirely: ``step`` always
    returns ``False`` and no state is tracked.
    """

    patience: int = 0
    min_delta: float = 0.0

    _best: float = field(default=math.inf, init=False, repr=False)
    _waiting: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.patience < 0:
            raise ModelError("patience must be >= 0")
        if self.min_delta < 0 or not math.isfinite(self.min_delta):
            raise ModelError("min_delta must be finite and >= 0")

    @property
    def best(self) -> float:
        """Best (lowest) metric value seen so far; ``inf`` when disabled."""
        return self._best

    def step(self, value: float) -> bool:
        """Record one observation; return ``True`` when training should stop."""
        if not math.isfinite(value):
            raise ModelError("value must be finite")
        if self.patience == 0:
            return False
        if value < self._best - self.min_delta:
            self._best = value
            self._waiting = 0
        else:
            self._waiting += 1
        return self._waiting >= self.patience


__all__ = ["EarlyStopper"]
