"""Excitation design for open-loop motor identification.

See ``docs/motor_id_guide.md`` for the full acquisition protocol.  The
rules encoded here:

* several step pulses at *different* duty cycles so one pulse can be held
  out for validation;
* every pulse lasts at least ``4 * tau`` (we use 4 s >> tau ≈ 50 ms) so
  the steady state is reached and K is well determined;
* rest periods between pulses so each pulse starts from standstill
  (also reveals the dead zone and friction offset);
* speeds during pulses should stay >= ``MIN_SAFE_SPEED_RPM`` so the
  encoder quantisation noise stays below ~1 %.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_SAFE_SPEED_RPM = 200.0          # speed needed for <2% quantisation noise
MIN_SAMPLES_PER_TIME_CONSTANT = 10  # fs must be >= 10 / tau


@dataclass(frozen=True)
class StepPulse:
    """One segment of the excitation sequence: constant PWM for a duration."""

    duration_s: float
    duty_pct: float


#: rest -> +20 % (train) -> rest -> +50 % (hold-out) -> rest
#: -> -50 % (train) -> rest
DEFAULT_SEQUENCE: tuple[StepPulse, ...] = (
    StepPulse(1.0, 0.0),
    StepPulse(4.0, 20.0),
    StepPulse(1.0, 0.0),
    StepPulse(4.0, 50.0),
    StepPulse(1.0, 0.0),
    StepPulse(4.0, -50.0),
    StepPulse(1.0, 0.0),
)


def sequence_to_input(sequence: tuple[StepPulse, ...], fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Expand a :class:`StepPulse` sequence into time and PWM arrays."""
    chunks_u = []
    for pulse in sequence:
        n = int(round(pulse.duration_s * fs))
        chunks_u.append(np.full(n, pulse.duty_pct, dtype=float))
    u = np.concatenate(chunks_u)
    t = np.arange(len(u), dtype=float) / fs
    return t, u


def prbs(n: int, levels: tuple[float, float], p_switch: float = 0.1,
         seed: int | None = None) -> np.ndarray:
    """Pseudo-random binary signal between two PWM levels (broadband excitation)."""
    if not 0.0 < p_switch < 1.0:
        raise ValueError("p_switch must be in (0, 1)")
    rng = np.random.default_rng(seed)
    u = np.empty(n)
    current = levels[0]
    for k in range(n):
        if rng.random() < p_switch:
            current = levels[1] if current == levels[0] else levels[0]
        u[k] = current
    return u


def recommend_sample_rate(bandwidth_hz: float) -> float:
    """Sample rate rule-of-thumb: >= 10x the closed-loop/plant bandwidth."""
    return float(max(100.0, MIN_SAMPLES_PER_TIME_CONSTANT * bandwidth_hz))
