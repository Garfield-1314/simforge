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


# --------------------------------------------------------------------------- #
# generic synthetic systems (for tests, demos and algorithm self-checks)
# --------------------------------------------------------------------------- #
def simulate_lti(u: np.ndarray, ad, bd, cd, dd, delay: int = 0,
                 meas_noise: float = 0.0, seed: int | None = None
                 ) -> np.ndarray:
    """Simulate a discrete state-space system with input delay + measurement noise."""
    u = np.asarray(u, dtype=float)
    n = len(u)
    ad = np.atleast_2d(np.asarray(ad, dtype=float))
    bd = np.atleast_2d(np.asarray(bd, dtype=float))
    cd = np.atleast_2d(np.asarray(cd, dtype=float))
    rng = np.random.default_rng(seed)
    nx = ad.shape[0]
    x = np.zeros(nx)
    y = np.empty(n)
    for k in range(n):
        uk = u[k - delay] if k >= delay else 0.0
        y[k] = (cd @ x).item()
        if meas_noise:
            y[k] += rng.normal(0.0, meas_noise)
        x = ad @ x + bd.reshape(nx) * uk
    return y


def second_order_system(fn_hz: float, zeta: float, gain: float = 1.0,
                        ts: float = 0.002):
    """Continuous G(s) = gain*wn^2/(s^2+2 z wn s+wn^2) -> discrete (ZOH).

    Returns (ad, bd, cd, dd, truth) with truth = dict(wn, zeta, gain).
    """
    from scipy.signal import cont2discrete
    wn = 2.0 * np.pi * fn_hz
    A = np.array([[0.0, 1.0], [-wn ** 2, -2.0 * zeta * wn]])
    B = np.array([[0.0], [wn ** 2]])
    C = np.array([[gain, 0.0]])
    D = np.array([[0.0]])
    ad, bd, cd, dd, _ = cont2discrete((A, B, C, D), ts)
    return ad, bd, cd, dd, {"wn": wn, "zeta": zeta, "gain": gain,
                            "fn_hz": fn_hz}


def generate_second_order_dataset(path: str | Path | None = None,
                                  fn_hz: float = 2.0, zeta: float = 0.4,
                                  gain: float = 2.0, ts: float = 0.002,
                                  delay: int = 3, meas_noise: float = 0.02,
                                  dwell: int = 50, n: int = 8000,
                                  seed: int = 7) -> tuple[SystemData | None, dict]:
    """Build (and optionally save) a second-order SISO dataset.

    Excitation is :func:`simforge.id.excitation.random_dwell` (broadband)
    so that order 2 is persistently excited.  Returns (data, truth).
    """
    from .excitation import random_dwell
    u = random_dwell(n, -10.0, 10.0, dwell, seed=seed)
    ad, bd, cd, dd, truth = second_order_system(fn_hz, zeta, gain, ts)
    y = simulate_lti(u, ad, bd, cd, dd, delay=delay, meas_noise=meas_noise,
                     seed=seed)
    truth = dict(truth, delay=delay, ts=ts, meas_noise=meas_noise,
                 excitation="random_dwell")
    if path is None:
        return None, truth
    data = SystemData(t=np.arange(n) * ts, u=u, y=y, fs=1.0 / ts,
                      input_unit="u", output_unit="y",
                      meta={"generator": "simforge.id.simulator",
                            "system": f"second_order fn={fn_hz}Hz zeta={zeta}",
                            "delay_samples": str(delay)})
    save_motor_csv(path, data)          # generic t_s,u,y layout (raw is None)
    from .io import load_system_csv
    return load_system_csv(path), truth
