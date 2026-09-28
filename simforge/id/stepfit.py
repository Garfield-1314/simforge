"""Output-error (OE) PEM fit of ``K * exp(-td*s) / (tau*s + 1)`` on speed.

Why the velocity domain and not cumulative counts:  integrating the
speed turns the (white) process disturbance into a random walk that
dominates the count residual, letting a 3-parameter optimiser "explain"
the drift instead of the dynamics.  At our operating range (>= 200 rpm,
1024x4 counts/rev, 500 Hz) the encoder quantisation of the per-period
pulse count is only +/-3.7 rpm (<= 2%), so the direct speed fit is both
simple and statistically sound:

    minimise  || v_model(u; K, tau, td) - v_measured ||^2   over train rows

with exact ZOH discretisation and fractional-delay input interpolation:

    v[k+1] = a*v[k] + K*(1-a)*u(t_k - td),   a = exp(-Ts/tau), v[0] = 0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter


@dataclass
class PemResult:
    K: float
    tau: float
    td: float
    ts: float
    K_ci: tuple[float, float]
    tau_ci: tuple[float, float]
    td_ci: tuple[float, float]
    rss: float
    n_rows: int
    success: bool
    message: str
    nfev: int

    @property
    def bandwidth_hz(self) -> float:
        return 1.0 / (2.0 * np.pi * self.tau)

    def simulate_velocity(self, u: np.ndarray) -> np.ndarray:
        """Free-run velocity response [rpm] of the fitted model."""
        return model_velocity((self.K, self.tau, self.td), u, self.ts)

    def to_dict(self) -> dict:
        return {
            "form": "K * exp(-td*s) / (tau*s + 1)",
            "K_rpm_per_pct": self.K,
            "tau_s": self.tau,
            "td_s": self.td,
            "bandwidth_hz": self.bandwidth_hz,
            "conf_95": {"K": list(self.K_ci), "tau": list(self.tau_ci),
                        "td": list(self.td_ci)},
            "fit": {"rss": self.rss, "n_rows": self.n_rows,
                    "converged": self.success, "nfev": self.nfev},
        }


def model_velocity(params, u: np.ndarray, ts: float) -> np.ndarray:
    """Velocity response [rpm] of the first-order model to input u (%).

    Exact zero-order-hold discretisation; the pure delay td is applied
    with linear interpolation so sub-sample delays are representable.
    """
    K, tau, td = params
    u = np.asarray(u, dtype=float)
    n = len(u)
    t = np.arange(n) * ts
    u_del = np.interp(t - td, t, u, left=u[0], right=0.0)
    a = float(np.exp(-ts / max(tau, 1e-6)))
    b = K * (1.0 - a)
    # y[k] = a*y[k-1] + b*u_del[k-1], y[0] = 0
    return lfilter([0.0, b], [1.0, -a], u_del)


def initial_guess(u: np.ndarray, v: np.ndarray,
                  tau0: float = 0.05, td0: float = 0.004) -> np.ndarray:
    """K from peak speed / peak duty, plus generic tau/td starting point."""
    max_duty = float(np.max(np.abs(u))) if len(u) else 0.0
    peak = float(np.max(np.abs(v))) if len(v) else 0.0
    K0 = peak / max_duty if max_duty > 1e-9 and peak > 1e-9 else 50.0
    return np.array([float(np.clip(K0, 1e-3, 1e4)), tau0, td0])


def identify_pem(u: np.ndarray, v: np.ndarray, ts: float,
                 train_idx: np.ndarray | None = None,
                 x0: np.ndarray | None = None) -> PemResult:
    """Nonlinear least-squares OE fit of (K, tau, td) to measured speed.

    ``train_idx`` restricts the residual to the training samples so that
    held-out segments remain a genuine out-of-sample test (the model is
    still simulated with the full input sequence).
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if len(u) != len(v):
        raise ValueError("u and v must have equal length")
    if train_idx is None:
        train_idx = np.arange(len(u))
    train_idx = np.asarray(train_idx, dtype=int)
    if x0 is None:
        x0 = initial_guess(u, v)

    def residual(params):
        if params[1] <= 1e-6 or params[2] < 0:
            return np.full(len(train_idx), 1e9)
        y = model_velocity(params, u, ts)
        return y[train_idx] - v[train_idx]

    sol = least_squares(
        residual, x0,
        bounds=([1e-6, 1e-4, 0.0], [1e5, 10.0, 0.1]),
        method="trf", x_scale="jac", ftol=1e-12, xtol=1e-12, max_nfev=500,
    )
    K, tau, td = (float(x) for x in sol.x)
    rss = float(2.0 * sol.cost)  # cost = 0.5 * sum(r^2)

    dof = max(len(train_idx) - 3, 1)
    se = np.full(3, np.nan)
    try:
        cov = np.linalg.pinv(sol.jac.T @ sol.jac) * (rss / dof)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except Exception:
        pass
    K_ci = (max(K - 1.96 * se[0], 0.0), K + 1.96 * se[0])
    tau_ci = (max(tau - 1.96 * se[1], 1e-6), tau + 1.96 * se[1])
    td_ci = (max(td - 1.96 * se[2], 0.0), td + 1.96 * se[2])

    return PemResult(K=K, tau=tau, td=td, ts=ts,
                     K_ci=K_ci, tau_ci=tau_ci, td_ci=td_ci,
                     rss=rss, n_rows=len(train_idx), success=bool(sol.success),
                     message=str(sol.message), nfev=int(sol.nfev))
