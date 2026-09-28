"""ARX(1,1) system identification with pure time delay.

Discrete model (velocity domain)::

    v[k+1] = a * v[k] + b * u[k - d] + e[k+1]

Least squares is linear in (a, b); the delay ``d`` is selected by AIC
over a small grid.  Continuous-time parameters follow from the exact
zero-order-hold discretisation of ``K / (tau*s + 1)``::

    a = exp(-Ts/tau)      b = K * (1 - a)
    =>  K = b / (1 - a),  tau = -Ts / ln(a),  t_d = d * Ts
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ArxResult:
    a: float
    b: float
    delay: int                 # delay in samples
    ts: float
    K: float                    # rpm per % PWM
    tau: float                  # s
    se_a: float
    se_b: float
    K_ci: tuple[float, float]
    tau_ci: tuple[float, float]
    rss: float
    n_rows: int
    aic: float
    resid: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    row_idx: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    delay_sweep: list[tuple[int, float]] = field(default_factory=list)

    @property
    def td(self) -> float:
        """Pure delay [s]."""
        return self.delay * self.ts

    @property
    def bandwidth_hz(self) -> float:
        """-3 dB bandwidth of K/(tau*s+1)."""
        return 1.0 / (2.0 * np.pi * self.tau)

    def simulate(self, u: np.ndarray, v0: float = 0.0) -> np.ndarray:
        """Free-run (open-loop) simulation of the discrete model."""
        u = np.asarray(u, dtype=float)
        n = len(u)
        v = np.empty(n)
        v[0] = v0
        d = self.delay
        for k in range(1, n):
            uk = u[k - 1 - d] if k - 1 - d >= 0 else 0.0
            v[k] = self.a * v[k - 1] + self.b * uk
        return v

    def one_step_residuals(self, v: np.ndarray, u: np.ndarray,
                           row_mask: np.ndarray | None = None) -> np.ndarray:
        """Innovation sequence e[k+1] = v[k+1] - a*v[k] - b*u[k-d]."""
        v = np.asarray(v, dtype=float)
        u = np.asarray(u, dtype=float)
        ks = np.arange(max(self.delay, 0), len(v) - 1)
        e = v[ks + 1] - self.a * v[ks] - self.b * u[ks - self.delay]
        if row_mask is not None:
            e = e[row_mask]
        return e

    def to_dict(self) -> dict:
        return {
            "form": "K * exp(-td*s) / (tau*s + 1)",
            "K_rpm_per_pct": self.K,
            "tau_s": self.tau,
            "td_s": self.td,
            "bandwidth_hz": self.bandwidth_hz,
            "discrete": {"a": self.a, "b": self.b, "delay_samples": self.delay},
            "std_err": {"a": self.se_a, "b": self.se_b},
            "conf_95": {"K": list(self.K_ci), "tau": list(self.tau_ci)},
            "fit": {"rss": self.rss, "n_rows": self.n_rows, "aic": self.aic},
        }


def _ls_fit(v: np.ndarray, u: np.ndarray, d: int):
    """Fit v[k+1] = a*v[k] + b*u[k-d] on rows whose outputs are in *valid*."""
    n = len(v)
    ks = np.arange(max(d, 0), n - 1)
    X = np.column_stack([v[ks], u[ks - d]])
    y = v[ks + 1]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    rss = float(resid @ resid)
    return ks, X, y, coef, rss


def identify_arx(u: np.ndarray, v: np.ndarray, ts: float,
                 max_delay: int | None = None,
                 exclude_rows: np.ndarray | None = None) -> ArxResult:
    """Identify ARX(1,1) with delay.

    Parameters
    ----------
    u, v
        Input (PWM %) and output (rpm) sequences.
    ts
        Sampling period [s].
    max_delay
        Largest delay to try, in samples (default: min(20, n/50)).
    exclude_rows
        Boolean-over-indices mask of *output* samples that must not be
        used (e.g. the hold-out region).  A row ``k -> k+1`` is dropped
        when either ``v[k]`` or ``v[k+1]`` is excluded.  Input samples
        are exogenous and never considered leakage.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if len(u) != len(v):
        raise ValueError("u and v must have equal length")
    n = len(v)
    if max_delay is None:
        max_delay = int(min(20, max(1, n // 50)))

    excluded = np.zeros(n, dtype=bool) if exclude_rows is None else np.asarray(exclude_rows, bool)

    best = None
    sweep: list[tuple[int, float]] = []
    for d in range(0, max_delay + 1):
        ks, X, y, coef, rss = _ls_fit(v, u, d)
        if exclude_rows is not None:
            keep = ~(excluded[ks] | excluded[ks + 1])
            if np.count_nonzero(keep) < max(10, 3 * n // 100):
                continue
            X, y = X[keep], y[keep]
            ks = ks[keep]
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ coef
            rss = float(resid @ resid)
        rows = len(y)
        if rows < 10:
            continue
        aic = rows * np.log(max(rss, 1e-300) / rows) + 2 * 3  # a, b, d
        sweep.append((d, aic))
        if best is None or aic < best[6]:
            best = (d, X, y, coef, rss, rows, aic, ks)

    if best is None:
        raise ValueError("ARX identification failed: not enough usable rows")

    d, X, y, coef, rss, rows, aic, ks_used = best
    a, b = float(coef[0]), float(coef[1])
    if not (0.0 < a < 1.0):
        raise ValueError(f"estimated pole a={a:.4f} outside (0,1); "
                         "data may be non-stationary or too short")

    dof = max(rows - 2, 1)
    sigma2 = rss / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se_a = float(np.sqrt(max(cov[0, 0], 0.0)))
    se_b = float(np.sqrt(max(cov[1, 1], 0.0)))

    K = b / (1.0 - a)
    tau = -ts / np.log(a)
    # delta method
    dK = np.array([b / (1.0 - a) ** 2, 1.0 / (1.0 - a)])
    se_K = float(np.sqrt(max(dK @ cov @ dK, 0.0)))
    dtau = ts / (a * np.log(a) ** 2)
    se_tau = float(abs(dtau) * se_a)

    z = 1.96
    K_ci = (max(K - z * se_K, 0.0), K + z * se_K)
    tau_ci = (max(tau - z * se_tau, 1e-6), tau + z * se_tau)

    resid = y - X @ coef
    return ArxResult(a=a, b=b, delay=int(d), ts=ts, K=K, tau=tau,
                     se_a=se_a, se_b=se_b, K_ci=K_ci, tau_ci=tau_ci,
                     rss=rss, n_rows=rows, aic=aic, resid=resid,
                     row_idx=ks_used, delay_sweep=sweep)
