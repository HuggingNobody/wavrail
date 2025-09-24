"""Energy-based voice activity detection.

The detector converts each frame to a mean-square level in dBFS and
compares it against ``threshold_db``. ``hangover_frames`` extends
every voiced region forward by up to N frames so short pauses inside
an utterance do not chop it up; the extension never makes a region
shorter than the raw threshold decision.

:class:`EnergyVad` keeps its state in a single counter so the same
policy can run frame-by-frame in a stream or in one batch pass;
:func:`detect_voice_activity` is the batch convenience wrapper. The
streaming and batch paths are contractually equivalent, and the
tests pin that invariant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wavrail.errors import FeatureError


@dataclass
class EnergyVad:
    """Frame-level voice activity detector with hangover.

    Attributes
    ----------
    threshold_db:
        Mean-square level in dBFS at or above which a frame counts
        as voiced.
    hangover_frames:
        Number of additional frames kept voiced after the level
        drops below the threshold.
    """

    threshold_db: float = -40.0
    hangover_frames: int = 2
    _hangover_left: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.hangover_frames < 0:
            raise FeatureError("hangover_frames must be non-negative")

    def reset(self) -> None:
        """Forget any pending hangover from a previous utterance."""
        self._hangover_left = 0

    def process_frame(self, frame: np.ndarray) -> bool:
        """Consume one frame and return whether it counts as voiced."""
        samples = np.asarray(frame, dtype=np.float64).reshape(-1)
        if samples.size == 0:
            raise FeatureError("frames must contain at least one sample")
        mean_square = float(np.mean(samples * samples))
        level_db = 10.0 * np.log10(mean_square) if mean_square > 0.0 else -np.inf
        if level_db >= self.threshold_db:
            self._hangover_left = self.hangover_frames
            return True
        if self._hangover_left > 0:
            self._hangover_left -= 1
            return True
        return False


def detect_voice_activity(
    frames: np.ndarray,
    threshold_db: float = -40.0,
    hangover_frames: int = 2,
) -> np.ndarray:
    """Return a boolean array with one voice-activity flag per frame."""
    arr = np.asarray(frames)
    if arr.ndim != 2:
        raise FeatureError("frames must be a 2-D array of shape (num_frames, frame_length)")
    vad = EnergyVad(threshold_db=threshold_db, hangover_frames=hangover_frames)
    return np.array([vad.process_frame(row) for row in arr], dtype=bool)


__all__ = ["EnergyVad", "detect_voice_activity"]
