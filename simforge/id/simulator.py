"""Synthetic brushed-DC motor data generator.

Produces a CSV with the *same* format the real experiment will use, from
known ground-truth parameters, so the whole identification pipeline can
be exercised (and its accuracy checked) before real data exists.

Physics of the generator
------------------------
* first-order speed dynamics  dn/dt = (K*u - n)/tau   (exact ZOH update)
* pure delay td on the input (PWM update + current build-up)
* optional process disturbance (white, rpm) -- load/voltage fluctuation
* quantisation: each sample records ``round(n * cpr/60 * Ts)`` pulses,
  exactly like a 1024-line x4 encoder at 500 Hz
* optional dead zone (PWM below ``deadzone`` produces no torque)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .excitation import DEFAULT_SEQUENCE, sequence_to_input
from .io import SystemData, counts_to_speed_rpm, save_motor_csv

#: ground truth used for the demo dataset (12 V, ~5000 rpm no-load)
TRUE_K = 50.0        # rpm per % PWM
TRUE_TAU = 0.05      # s
TRUE_TD = 0.004      # s


def simulate_motor(u: np.ndarray, fs: float, K: float = TRUE_K,
                   tau: float = TRUE_TAU, td: float = TRUE_TD,
                   cpr: int = 4096, noise_rpm: float = 15.0,
                   deadzone_pct: float = 0.0, seed: int | None = 42
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate speed and per-period encoder counts for PWM input u.

    Returns (t, true_speed_rpm, enc_delta).
    """
    u = np.asarray(u, dtype=float)
    ts = 1.0 / fs
    n = len(u)
    rng = np.random.default_rng(seed)

    # dead zone + input delay
    u_eff = np.sign(u) * np.maximum(np.abs(u) - deadzone_pct, 0.0)
    t = np.arange(n) * ts
    u_del = np.interp(t - td, t, u_eff, left=u_eff[0], right=0.0)

    a = float(np.exp(-ts / tau))
    b = K * (1.0 - a)
    speed = np.empty(n)
    speed[0] = 0.0
    for k in range(1, n):
        speed[k] = a * speed[k - 1] + b * u_del[k - 1] + rng.normal(0.0, noise_rpm)

    enc_delta = np.rint(speed * ts * cpr / 60.0)
    return t, speed, enc_delta


def generate_dataset(path: str | Path, fs: float = 500.0, cpr: int = 4096,
                     K: float = TRUE_K, tau: float = TRUE_TAU, td: float = TRUE_TD,
                     noise_rpm: float = 15.0, deadzone_pct: float = 0.0,
                     seed: int | None = 42,
                     sequence=DEFAULT_SEQUENCE) -> SystemData:
    """Simulate the default excitation sequence and write it as CSV.

    Returns the :class:`SystemData` re-loaded from disk so callers get the
    exact object the identification pipeline will consume.
    """
    t, u = sequence_to_input(sequence, fs)
    _, speed, enc = simulate_motor(u, fs, K=K, tau=tau, td=td, cpr=cpr,
                                   noise_rpm=noise_rpm,
                                   deadzone_pct=deadzone_pct, seed=seed)
    data = SystemData(t=t, u=u, y=counts_to_speed_rpm(enc, fs, cpr), fs=fs,
                      input_unit="% PWM", output_unit="rpm",
                      meta={
                          "fs": f"{fs:g}",
                          "counts_per_rev": str(cpr),
                          "supply_V": "12",
                          "load": "wheel_suspended",
                          "encoder": "1024 lines x4 quadrature, no gear",
                          "generator": "simforge.id.simulator",
                          "true_k": f"{K:g}",
                          "true_tau_s": f"{tau:g}",
                          "true_td_s": f"{td:g}",
                          "noise_rpm": f"{noise_rpm:g}",
                          "deadzone_pct": f"{deadzone_pct:g}",
                          "seed": str(seed),
                      },
                      raw=enc)
    save_motor_csv(path, data)
    return load(path)


def load(path: str | Path) -> SystemData:
    from .io import load_motor_csv
    return load_motor_csv(path)
