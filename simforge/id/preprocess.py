"""Segmenting and index bookkeeping for multi-step experiment records."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import MotorData


@dataclass(frozen=True)
class Segment:
    """A constant-PWM stretch of the recording.

    ``idx0`` is the sample at which the PWM changed (start of segment),
    ``idx1`` the exclusive end (the sample of the *next* change).
    """

    idx0: int
    idx1: int
    duty: float

    @property
    def length(self) -> int:
        return self.idx1 - self.idx0

    @property
    def duration(self) -> float:
        return self.length  # caller scales by ts


def find_segments(data: MotorData, min_step: float = 1.0) -> list[Segment]:
    """Split the recording wherever |ΔPWM| >= *min_step* percent."""
    changes = np.flatnonzero(np.abs(np.diff(data.pwm)) >= min_step) + 1
    bounds = [0, *changes.tolist(), data.n]
    segs = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end > start:
            segs.append(Segment(idx0=start, idx1=end, duty=float(data.pwm[start])))
    return segs


def pulse_segments(segments: list[Segment], eps: float = 1e-9) -> list[tuple[int, Segment]]:
    """Indices + segments where the PWM is non-zero (actual step pulses)."""
    return [(i, s) for i, s in enumerate(segments) if abs(s.duty) > eps]


def holdout_indices(data: MotorData, segments: list[Segment],
                    holdout_duty: float, tol: float = 1e-6) -> np.ndarray:
    """Sample indices of the held-out pulse *including* the rest after it.

    The model never sees these output samples during identification, so
    metrics computed on them are genuine out-of-sample tests.
    """
    for i, seg in enumerate(segments):
        if abs(seg.duty - holdout_duty) <= tol and abs(seg.duty) > tol:
            # extend through the following rest segment (if any)
            end = seg.idx1
            if i + 1 < len(segments) and abs(segments[i + 1].duty) <= tol:
                end = segments[i + 1].idx1
                if i + 2 < len(segments):
                    end = segments[i + 2].idx0
            end = min(max(end, seg.idx1), data.n)
            return np.arange(seg.idx0, end)
    raise ValueError(f"hold-out duty {holdout_duty} % not found in recording")


def complement_indices(n: int, excluded: np.ndarray) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[excluded] = True
    return np.flatnonzero(~mask)


def steady_window(seg: Segment, settle_frac: float = 0.5) -> np.ndarray:
    """Last *settle_frac* of a pulse -- assumed to be in steady state."""
    start = seg.idx0 + int(round((1.0 - settle_frac) * seg.length))
    return np.arange(start, seg.idx1)
