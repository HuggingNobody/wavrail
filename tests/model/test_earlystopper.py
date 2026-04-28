"""EarlyStopper patience and min_delta logic (no torch required)."""

from __future__ import annotations

import math

import pytest

from wavrail.errors import ModelError
from wavrail.model import EarlyStopper

pytestmark = pytest.mark.model


def test_patience_counts_non_improving_steps():
    stopper = EarlyStopper(patience=2)
    assert not stopper.step(1.0)
    assert not stopper.step(0.5)  # improvement resets the counter
    assert not stopper.step(0.6)  # waiting = 1
    assert stopper.step(0.7)  # waiting = 2 -> stop


def test_min_delta_requires_meaningful_improvement():
    stopper = EarlyStopper(patience=1, min_delta=0.1)
    assert not stopper.step(1.0)
    assert stopper.step(0.95)  # 0.05 < 0.1 does not count as improvement


def test_disabled_never_stops():
    stopper = EarlyStopper(patience=0)
    for value in [1.0, 1.0, 1.0, 2.0]:
        assert not stopper.step(value)
    assert stopper.best == math.inf


def test_best_tracks_minimum():
    stopper = EarlyStopper(patience=3)
    for value in [2.0, 1.0, 1.5]:
        stopper.step(value)
    assert stopper.best == 1.0


def test_validation():
    with pytest.raises(ModelError):
        EarlyStopper(patience=-1)
    with pytest.raises(ModelError):
        EarlyStopper(min_delta=-0.1)
    with pytest.raises(ModelError):
        EarlyStopper(min_delta=float("inf"))
    with pytest.raises(ModelError):
        EarlyStopper(patience=1).step(float("nan"))
