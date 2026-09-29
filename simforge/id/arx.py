"""ARX(n,m) system identification with pure time delay.

Linear least squares over a grid of orders (n, m) and delays d, with
model-order selection by AIC/BIC.  Discrete model::

    y[k+1] = sum_{i=0..n-1} a_i y[k-i]  +  sum_{j=0..m-1} b_j u[k-d-j] + e

For n = m = 1 this is exactly the original first-order motor model
``v[k+1] = a v[k] + b u[k-d]`` with::

    K = b / (1 - a),   tau = -Ts / ln(a),   t_d = d * Ts

Higher orders expose the full transfer function through
:class:`simforge.id.tf.TransferFunction` (G(z), G(s), characteristic
equation, poles/zeros).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tf import TransferFunction


@dataclass
class ArxResult:
    a_vec: np.ndarray                 # recursion coefficients a_0..a_{n-1}
    b_vec: np.ndarray                 # input taps b_0..b_{m-1}
    delay: int                        # d (samples); y[k+1] uses u[k-d]
    ts: float
    se_a: np.ndarray
    se_b: np.ndarray
    K: float                          # DC gain (output unit / input unit)
    K_ci: tuple[float, float]
    tau: float | None                 # order-1 exact, else dominant pole tau
    tau_ci: tuple[float, float] | None
    rss: float
    n_rows: int
    aic: float
    criterion: str = "bic"
    input_unit: str = ""
    output_unit: str = ""
    resid: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    row_idx: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    sweep: list[tuple[int, int, int, float]] = field(default_factory=list)
    order_sweep: dict[int, float] = field(default_factory=dict)
    method: str = "auto"

    # ---- compat scalar accessors (first coefficients) -------------------- #
    @property
    def a(self) -> float:
        return float(self.a_vec[0])

    @property
    def b(self) -> float:
        return float(self.b_vec[0])

    @property
    def n_order(self) -> int:
        return len(self.a_vec)

    @property
    def m_order(self) -> int:
        return len(self.b_vec)

    @property
    def td(self) -> float:
        """Pure delay parameter [s]."""
        return self.delay * self.ts

    @property
    def bandwidth_hz(self) -> float | None:
        return None if self.tau is None else 1.0 / (2.0 * np.pi * self.tau)

    @property
    def delay_sweep(self) -> list[tuple[int, float]]:
        """(compat) per-delay AIC of the first-order models."""
        return [(d, aic) for (n, m, d, aic) in self.sweep if n == 1 and m == 1]

    @property
    def order_gap(self) -> float | None:
        """AIC gap between the chosen order and the best *other* order.

        Large gap (>= ~10) => the selected order is well supported;
        small gap => order uncertainty worth reporting.
        """
        if len(self.order_sweep) < 2:
            return None
        best = self.aic
        others = [v for k, v in self.order_sweep.items() if k != self.n_order]
        return float(min(others) - best)

    def to_tf(self) -> TransferFunction:
        return TransferFunction(a=self.a_vec, b=self.b_vec, delay=self.delay,
                                ts=self.ts, input_unit=self.input_unit,
                                output_unit=self.output_unit)

    # ---- simulation / residuals ------------------------------------------ #
    def simulate(self, u: np.ndarray, y0: float = 0.0) -> np.ndarray:
        """Free-run (open-loop) simulation of the model."""
        from .stepfit import simulate_oe
        return simulate_oe(self.a_vec, self.b_vec, self.delay,
                           np.asarray(u, dtype=float), y0=y0)

    def regressors(self, y: np.ndarray, u: np.ndarray, ks: np.ndarray) -> np.ndarray:
        X = np.column_stack(
            [y[ks - i] for i in range(len(self.a_vec))]
            + [u[ks - self.delay - j] for j in range(len(self.b_vec))]
        )
        return X

    def innovations(self, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        """One-step prediction errors on exactly the fitted rows."""
        y = np.asarray(y, dtype=float)
        u = np.asarray(u, dtype=float)
        ks = self.row_idx
        X = self.regressors(y, u, ks)
        return y[ks + 1] - X @ np.concatenate([self.a_vec, self.b_vec])

    # ---- serialisation ----------------------------------------------------- #
    def to_dict(self) -> dict:
        tf = self.to_tf()
        out = {
            "form": "ARX(n,m) + pure delay  (y[k+1] = A y[k] + B u[k-d])",
            "order": {"n": self.n_order, "m": self.m_order},
            "estimator": self.method,
            "dc_gain": self.K,
            "delay_samples": self.delay,
            "td_s": self.td,
            "discrete": {"a": [float(x) for x in self.a_vec],
                         "b": [float(x) for x in self.b_vec],
                         "delay_samples": int(self.delay)},
            "std_err": {"a": [float(x) for x in self.se_a],
                        "b": [float(x) for x in self.se_b]},
            "conf_95": {"dc_gain": list(self.K_ci)},
            "fit": {"rss": self.rss, "n_rows": self.n_rows,
                    self.criterion: self.aic},
            "transfer_function": tf.to_dict(),
        }
        # scalar time-constant view (exact for order 1, dominant pole else)
        if self.tau is not None:
            out["tau_s"] = self.tau
            out["bandwidth_hz"] = self.bandwidth_hz
        if self.tau_ci is not None:
            out["conf_95"]["tau"] = list(self.tau_ci)
        return out


# --------------------------------------------------------------------------- #
def _build_rows(y: np.ndarray, u: np.ndarray, n: int, m: int, d: int):
    """Regressor matrices predicting y[k+1] from y[k..], u[k-d..]."""
    kmin = max(n - 1, d + m - 1)
    ks = np.arange(kmin, len(y) - 1)
    if len(ks) < 10:
        return None
    X = np.column_stack(
        [y[ks - i] for i in range(n)] + [u[ks - d - j] for j in range(m)]
    )
    target = y[ks + 1]
    return ks, X, target


def _fit_combo(y, u, n, m, d, excluded, criterion, method="iv"):
    """Fit one (n, m, d) combination.

    ``method="ls"`` is plain least squares -- consistent when the additive
    disturbance enters the *state* (process noise) but biased towards
    smaller poles when the output carries measurement noise (the noise
    sits in the regressors; errors-in-variables attenuation).

    ``method="iv"`` instruments the regressors with lagged inputs
    ``u[k-r]`` (exogenous, orthogonal to any measurement noise), which
    keeps estimates consistent and prevents noise-fitting from biasing
    the order/delay search.  Returns None when the fit is impossible.
    """
    built = _build_rows(y, u, n, m, d)
    if built is None:
        return None
    ks, X, target = built
    if excluded is not None:
        keep = ~(excluded[ks] | excluded[ks + 1])
        if np.count_nonzero(keep) < max(10, 3 * len(y) // 100):
            return None
        X, target, ks = X[keep], target[keep], ks[keep]
    rows = len(target)
    if rows < 10:
        return None

    Z = None
    if method == "iv":
        # Instrument lag window: short lags around the input taps PLUS a
        # log-spaced grid spanning the record.  Log spacing matters: a
        # short window (a few samples) is nearly collinear during input
        # plateaus and cannot instrument slow dynamic directions.
        small = list(range(0, d + m + 2))
        max_lag = int(np.clip(len(y) // 10, 40, 2000))
        log_grid = (np.round(np.logspace(np.log10(3), np.log10(max_lag), 10))
                    .astype(int).tolist()) if max_lag > 6 else []
        lags = sorted(set(small) | set(int(r) for r in log_grid))
        msk = ks >= lags[-1]
        if np.count_nonzero(msk) < max(10, 3 * rows // 100):
            return None
        Xs, ys, kss = X[msk], target[msk], ks[msk]
        Z = np.column_stack([u[kss - r] for r in lags])
        coef, *_ = np.linalg.lstsq(Z.T @ Xs, Z.T @ ys, rcond=None)
        if not np.all(np.isfinite(coef)):
            return None
        X, target, ks = Xs, ys, kss
        rows = len(target)
    else:
        coef, *_ = np.linalg.lstsq(X, target, rcond=None)

    resid = target - X @ coef
    rss = float(resid @ resid)
    p = n + m + 1                              # coefficients + delay
    pen = 2.0 if criterion == "aic" else np.log(max(rows, 2))
    crit = rows * np.log(max(rss, 1e-300) / rows) + pen * p
    return ks, X, target, coef, rss, rows, crit, Z


def _finalize_fit(fit, ts, method, criterion, sweep=None, order_best=None,
                  input_unit="", output_unit="") -> ArxResult:
    """Post-process a fitted (n, m, d) combo into an :class:`ArxResult`."""
    n, m, d, ks, X, target, coef, rss, rows, crit, Z = fit
    a_vec = np.asarray(coef[:n], dtype=float)
    b_vec = np.asarray(coef[n:], dtype=float)

    # ---- stability guard ------------------------------------------------- #
    tf = TransferFunction(a=a_vec, b=b_vec, delay=d, ts=ts,
                          input_unit=input_unit, output_unit=output_unit)
    if not tf.is_stable:
        raise ValueError(
            f"estimated model is unstable (poles {np.round(tf.poles_z(), 4)}); "
            "data may be non-stationary, too short or too noisy")

    # ---- covariance / confidence intervals -------------------------------- #
    dof = max(rows - (n + m), 1)
    sigma2 = rss / dof
    if method == "iv" and Z is not None:
        M = Z.T @ X
        Minv = np.linalg.pinv(M)
        cov = sigma2 * (Minv @ (Z.T @ Z) @ Minv.T)
    else:
        cov = sigma2 * np.linalg.pinv(X.T @ X)
    cov = np.atleast_2d(cov)
    se_a = np.sqrt(np.clip(np.diag(cov)[:n], 0.0, None))
    se_b = np.sqrt(np.clip(np.diag(cov)[n:], 0.0, None))

    S = 1.0 - float(np.sum(a_vec))
    B = float(np.sum(b_vec))
    if abs(S) < 1e-12:
        raise ValueError("degenerate DC gain (1 - sum(a) == 0)")
    K = B / S
    # delta method: dK/da_i = K/S ; dK/db_j = 1/S
    g = np.concatenate([np.full(n, K / S), np.full(m, 1.0 / S)])
    se_K = float(np.sqrt(max(g @ cov @ g, 0.0)))
    K_ci = (max(K - 1.96 * se_K, 0.0), K + 1.96 * se_K)

    # tau: exact for first-order models, dominant pole otherwise
    tau: float | None = None
    tau_ci: tuple[float, float] | None = None
    if n == 1 and 0.0 < a_vec[0] < 1.0:
        tau = float(-ts / np.log(a_vec[0]))
        dtau = ts / (a_vec[0] * np.log(a_vec[0]) ** 2)
        se_tau = float(abs(dtau) * se_a[0])
        tau_ci = (max(tau - 1.96 * se_tau, 1e-6), tau + 1.96 * se_tau)
    else:
        tau = tf.tau_eff()

    resid = target - X @ np.concatenate([a_vec, b_vec])
    return ArxResult(
        a_vec=a_vec, b_vec=b_vec, delay=int(d), ts=ts,
        se_a=se_a, se_b=se_b, K=float(K), K_ci=K_ci,
        tau=tau, tau_ci=tau_ci, rss=rss, n_rows=rows, aic=float(crit),
        criterion=criterion, input_unit=input_unit, output_unit=output_unit,
        resid=resid, row_idx=ks, sweep=sweep if sweep is not None else [],
        order_sweep=order_best if order_best is not None else {},
        method=method,
    )


def identify_arx_structure(u: np.ndarray, y: np.ndarray, ts: float,
                           n: int, m: int, d: int,
                           exclude_rows: np.ndarray | None = None,
                           criterion: str = "bic", method: str = "iv",
                           input_unit: str = "", output_unit: str = "") -> ArxResult:
    """Fit one fixed ARX(n, m, d) structure (used by PEM-based selection)."""
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    excluded = None if exclude_rows is None else np.asarray(exclude_rows, bool)
    fit = _fit_combo(y, u, n, m, d, excluded, criterion, method)
    if fit is None:
        raise ValueError(f"cannot fit ARX({n},{m},d={d}): not enough usable rows")
    return _finalize_fit((n, m, d) + fit, ts, method, criterion,
                         input_unit=input_unit, output_unit=output_unit)


#: relative LS-vs-IV coefficient gap that indicates measurement noise on y
#: (below -> plain LS is efficient; above -> IV is needed for consistency)
AUTO_METHOD_DIFF = 0.10


def fit_structure_auto(u: np.ndarray, y: np.ndarray, ts: float,
                       n: int, m: int, d: int,
                       exclude_rows: np.ndarray | None = None,
                       criterion: str = "bic",
                       input_unit: str = "", output_unit: str = ""
                       ) -> tuple[ArxResult, dict]:
    """Fit one structure with both LS and IV and pick the appropriate one.

    The LS/IV discrepancy is a practical measurement-vs-process noise
    detector: both estimators agree (small gap) when the disturbance
    enters the state -- plain LS is then the low-variance choice; they
    disagree (large gap) when the output carries measurement noise,
    where only IV is consistent.
    """
    fits: dict = {}
    for meth in ("ls", "iv"):
        try:
            fits[meth] = identify_arx_structure(
                u, y, ts, n, m, d, exclude_rows=exclude_rows,
                criterion=criterion, method=meth,
                input_unit=input_unit, output_unit=output_unit)
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not fits:
        raise ValueError(f"cannot fit ARX({n},{m},d={d}) with either method")
    if len(fits) == 1:
        meth, res = next(iter(fits.items()))
        return res, {"auto_method": meth, "auto_diff": float("nan")}
    ls_res, iv_res = fits["ls"], fits["iv"]
    # per-group relative disagreement: normalise a- and b-coefficients by
    # their own group scale so a large a0 cannot mask a b-tap error
    def _group_rel(dv: np.ndarray, ref: np.ndarray) -> float:
        scale = float(np.max(np.abs(ref))) if len(ref) else 0.0
        if scale <= 0:
            return 0.0
        denom = np.maximum(np.abs(ref), 0.1 * scale)
        return float(np.max(np.abs(dv) / denom))

    diff = max(_group_rel(ls_res.a_vec - iv_res.a_vec, iv_res.a_vec),
               _group_rel(ls_res.b_vec - iv_res.b_vec, iv_res.b_vec))
    chosen = "ls" if diff < AUTO_METHOD_DIFF else "iv"
    return fits[chosen], {"auto_method": chosen, "auto_diff": diff}


def identify_arx_auto(u: np.ndarray, y: np.ndarray, ts: float,
                      max_order: int = 3, max_delay: int | None = None,
                      criterion: str = "bic", method: str = "auto",
                      exclude_rows: np.ndarray | None = None,
                      input_unit: str = "", output_unit: str = "") -> ArxResult:
    """Identify an ARX(n,m) model, selecting (n, m, d) by AIC/BIC.

    Parameters
    ----------
    max_order
        Largest denominator order n to consider; m runs 1..n.
    max_delay
        Largest delay to try, in samples (default min(20, N//50)).
    criterion
        ``"bic"`` (default -- ln(N)*params, robust against over-fitting
        order on long records) or ``"aic"`` (2*params).
    method
        ``"auto"`` (default -- see :func:`fit_structure_auto`: the grid
        runs with IV, then the winning structure is refit with LS and IV
        and the appropriate estimator is chosen by their discrepancy),
        ``"iv"`` (instrumental variables, consistent under output
        measurement noise) or ``"ls"`` (plain least squares, efficient
        under process noise).
    exclude_rows
        Boolean mask of *output* samples withheld from identification
        (hold-out region).  A row k->k+1 is dropped when either y[k] or
        y[k+1] is excluded; inputs are exogenous and never leakage.
    """
    if criterion not in ("aic", "bic"):
        raise ValueError("criterion must be 'aic' or 'bic'")
    if method not in ("auto", "iv", "ls"):
        raise ValueError("method must be 'auto', 'iv' or 'ls'")
    grid_method = "iv" if method == "auto" else method
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(u) != len(y):
        raise ValueError("u and y must have equal length")
    n_total = len(y)
    if max_order < 1:
        raise ValueError("max_order must be >= 1")
    if max_delay is None:
        max_delay = int(min(20, max(1, n_total // 50)))
    excluded = None if exclude_rows is None else np.asarray(exclude_rows, bool)

    best = None
    sweep: list[tuple[int, int, int, float]] = []
    order_best: dict[int, float] = {}
    for n in range(1, max_order + 1):
        for m in range(1, n + 1):
            for d in range(0, max_delay + 1):
                fit = _fit_combo(y, u, n, m, d, excluded, criterion, grid_method)
                if fit is None:
                    continue
                ks, X, target, coef, rss, rows, crit, Z = fit
                sweep.append((n, m, d, crit))
                if n not in order_best or crit < order_best[n]:
                    order_best[n] = crit
                if best is None or crit < best[9]:
                    best = (n, m, d, ks, X, target, coef, rss, rows, crit, Z)

    if best is None:
        raise ValueError("ARX identification failed: not enough usable rows")

    n_best, m_best, d_best = best[0], best[1], best[2]
    if method == "auto":
        res, _info = fit_structure_auto(
            u, y, ts, n_best, m_best, d_best, exclude_rows=exclude_rows,
            criterion=criterion, input_unit=input_unit,
            output_unit=output_unit)
        res.sweep = sweep
        res.order_sweep = order_best
        return res
    return _finalize_fit(best, ts, method, criterion, sweep=sweep,
                         order_best=order_best, input_unit=input_unit,
                         output_unit=output_unit)


def identify_arx(u: np.ndarray, y: np.ndarray, ts: float,
                 max_delay: int | None = None,
                 exclude_rows: np.ndarray | None = None,
                 **kwargs) -> ArxResult:
    """First-order ARX(1,1) + delay (compat wrapper around auto-order fit)."""
    return identify_arx_auto(u, y, ts, max_order=1, max_delay=max_delay,
                             exclude_rows=exclude_rows, **kwargs)


def arx_result_from_coeffs(a, b, delay: int, ts: float, **kwargs) -> ArxResult:
    """Build an ArxResult from known coefficients (used by tests/PEM init)."""
    a_vec = np.atleast_1d(np.asarray(a, dtype=float))
    b_vec = np.atleast_1d(np.asarray(b, dtype=float))
    tf = TransferFunction(a=a_vec, b=b_vec, delay=delay, ts=ts)
    S = 1.0 - float(np.sum(a_vec))
    K = float(np.sum(b_vec) / S)
    tau = (-ts / np.log(a_vec[0])) if (len(a_vec) == 1 and 0 < a_vec[0] < 1) \
        else tf.tau_eff()
    return ArxResult(a_vec=a_vec, b_vec=b_vec, delay=int(delay), ts=ts,
                     se_a=np.full(len(a_vec), np.nan),
                     se_b=np.full(len(b_vec), np.nan),
                     K=K, K_ci=(np.nan, np.nan), tau=tau, tau_ci=None,
                     rss=np.nan, n_rows=0, aic=np.nan, **kwargs)
