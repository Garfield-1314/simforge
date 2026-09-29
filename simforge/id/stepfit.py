"""Output-error (OE) PEM fit of an ARX-structured model to measured output.

The model structure (orders n, m and delay d) comes from the ARX fit; the
nonlinear least-squares search fits the coefficients (a, b) by simulating
the model and comparing *simulated* output with measurements on the
training rows only (held-out data stays out-of-sample):

    minimise  || y_model(u; a, b) - y_measured ||^2   over train rows

The simulation uses the exact recursion

    y[k] = sum_i a_i y[k-1-i] + sum_j b_j u[k-1-d-j],   y[0] = y0

which for zero initial condition is evaluated as a stable IIR filter
(``scipy.signal.lfilter``); a sequential path handles non-zero starts.

Why velocity/output domain and not an integrated (counts) signal:
integrating turns the white process disturbance into a random walk that
dominates the residual, letting a low-dimensional optimiser "explain"
the drift instead of the dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter

from .tf import TransferFunction, first_order_from_arx


@dataclass
class PemResult:
    K: float                           # DC gain
    tau: float | None                  # order-1 exact, else dominant pole tau
    td: float
    ts: float
    a_vec: np.ndarray
    b_vec: np.ndarray
    delay: int
    K_ci: tuple[float, float]
    tau_ci: tuple[float, float] | None
    rss: float
    n_rows: int
    success: bool
    message: str
    nfev: int
    input_unit: str = ""
    output_unit: str = ""

    @property
    def bandwidth_hz(self) -> float | None:
        return None if self.tau is None else 1.0 / (2.0 * np.pi * self.tau)

    @property
    def n_order(self) -> int:
        return len(self.a_vec)

    def to_tf(self) -> TransferFunction:
        return TransferFunction(a=self.a_vec, b=self.b_vec, delay=self.delay,
                                ts=self.ts, input_unit=self.input_unit,
                                output_unit=self.output_unit)

    def simulate_velocity(self, u: np.ndarray) -> np.ndarray:
        """Free-run model response y_model(u)."""
        return simulate_oe(self.a_vec, self.b_vec, self.delay,
                           np.asarray(u, dtype=float), y0=0.0)

    def to_dict(self) -> dict:
        out = {
            "form": "OE-PEM of ARX(n,m) + pure delay",
            "order": {"n": self.n_order, "m": len(self.b_vec)},
            "dc_gain": self.K,
            "td_s": self.td,
            "discrete": {"a": [float(x) for x in self.a_vec],
                         "b": [float(x) for x in self.b_vec],
                         "delay_samples": int(self.delay)},
            "conf_95": {"dc_gain": list(self.K_ci)},
            "fit": {"rss": self.rss, "n_rows": self.n_rows,
                    "converged": self.success, "nfev": self.nfev},
            "transfer_function": self.to_tf().to_dict(),
        }
        # legacy aliases (first-order naming)
        out["K_rpm_per_pct"] = self.K
        if self.tau is not None:
            out["tau_s"] = self.tau
            out["bandwidth_hz"] = self.bandwidth_hz
            out["conf_95"]["tau"] = list(self.tau_ci) if self.tau_ci else []
        return out


# --------------------------------------------------------------------------- #
# structure selection (two-stage: ARX candidates -> consistent PEM/BIC)
# --------------------------------------------------------------------------- #
@dataclass
class StructureSelection:
    """Result of PEM-based model structure selection."""

    arx: ArxResult                       # IV refit at the chosen structure
    pem: PemResult                       # OE-refined coefficients
    table: list                          # all evaluated candidates
    order_bic: dict                      # best PEM-based criterion per order
    criterion: str = "bic"
    simplicity_window: float = 0.0       # criterion units of the 5%-RSS tie rule

    @property
    def n(self) -> int:
        return self.arx.n_order

    @property
    def m(self) -> int:
        return self.arx.m_order

    @property
    def delay(self) -> int:
        return self.arx.delay

    @property
    def best_value(self) -> float:
        return float(self.arx.aic)

    @property
    def order_gap(self) -> float:
        """Criterion gap between the *chosen* order and the best other order.

        Negative when the simplicity tie-break selected a lower order
        than the raw argmin -- i.e. the evidence for the extra dynamics
        was weak and the simpler model was preferred.
        """
        if len(self.order_bic) < 2 or self.n not in self.order_bic:
            return float("nan")
        others = [v for k, v in self.order_bic.items() if k != self.n]
        return float(min(others) - self.order_bic[self.n])


def select_structure_pem(u: np.ndarray, y: np.ndarray, ts: float,
                         arx_stage1, train_idx: np.ndarray | None = None,
                         exclude_rows: np.ndarray | None = None,
                         d_window: int = 2, criterion: str = "bic",
                         simplicity_rss_ratio: float = 1.05,
                         input_unit: str = "", output_unit: str = ""
                         ) -> StructureSelection:
    """Pick (n, m, d) by *output-error* criterion, then fit ARX + PEM.

    Why not plain ARX-BIC: with measurement noise on y the in-sample ARX
    residual rewards noise fitting differently for each structure
    (errors-in-variables), so the ARX criterion can prefer a wrong order
    or delay.  The OE (simulation-error) residual is consistent, so each
    ARX candidate structure is refined by :func:`identify_pem` and ranked
    by ``n*ln(RSS/n) + pen*p`` on the training rows only.

    ``arx_stage1`` is one :class:`ArxResult` or a list of them (e.g. the
    LS and IV grid searches); candidates are, for every (n, m), the union
    of each stage-1 best delay +/- ``d_window`` samples.  The winning
    structure is refit with :func:`~simforge.id.arx.fit_structure_auto`
    so the reported ARX coefficients/covariance use the
    noise-appropriate estimator.

    Simplicity tie-break: because the disturbance (colored, unmodelled)
    is not part of the criterion, extra poles can absorb a few percent
    of sample-specific disturbance projection and win on RSS alone.
    Orders whose criterion is within ``rows * ln(simplicity_rss_ratio)``
    of the best are treated as tied and the lowest such order wins.

    To keep noise-fitting from deciding, candidates are fitted on a
    random 70 % of the training rows and ranked by their *validation*
    residual on the held-out 30 % (noise projection does not
    generalise; genuine dynamics do).  The winning structure is then
    refitted on all training rows.
    """
    from .arx import identify_arx_structure

    if not isinstance(arx_stage1, (list, tuple)):
        arx_stage1 = [arx_stage1]

    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    if train_idx is None:
        train_idx = np.arange(len(u))
    train_idx = np.asarray(train_idx, dtype=int)
    simplicity_window = len(train_idx) * np.log(max(simplicity_rss_ratio, 1.0))

    # 70/30 fit/validation split of the training rows (fixed seed)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(train_idx))
    n_fit = max(int(0.7 * len(train_idx)), 10)
    fit_idx = train_idx[perm[:n_fit]]
    val_idx = train_idx[perm[n_fit:]]
    if len(val_idx) < 10:            # tiny datasets: rank on all rows
        fit_idx = val_idx = train_idx

    # best ARX delay per (n, m) from all stage-1 sweeps
    best_d: dict[tuple[int, int], set] = {}
    for st in arx_stage1:
        for (nn, mm, dd, _crit) in st.sweep:
            best_d.setdefault((nn, mm), set()).add(int(dd))
    if not best_d:
        raise ValueError("stage-1 ARX sweep is empty")

    pen_val = np.log(max(len(val_idx), 2))
    table: list = []
    order_bic: dict[int, float] = {}
    for (nn, mm), d0s in sorted(best_d.items()):
        cand_d: set = set()
        for d0 in d0s:
            cand_d.update(range(max(0, d0 - d_window), d0 + d_window + 1))
        for dd in sorted(cand_d):
            try:
                arx_c = identify_arx_structure(
                    u, y, ts, nn, mm, dd, exclude_rows=exclude_rows,
                    criterion=criterion, input_unit=input_unit,
                    output_unit=output_unit)
                pem_c = identify_pem(u, y, ts, train_idx=fit_idx,
                                     structure=arx_c, input_unit=input_unit,
                                     output_unit=output_unit)
            except (ValueError, np.linalg.LinAlgError):
                continue
            yh = pem_c.simulate_velocity(u)
            rss_val = float(np.sum((yh[val_idx] - y[val_idx]) ** 2))
            rows_v = max(len(val_idx), 2)
            p = nn + mm + 1
            pen = 2.0 if criterion == "aic" else pen_val
            crit_val = rows_v * np.log(max(rss_val, 1e-300) / rows_v) + pen * p
            table.append({"n": nn, "m": mm, "d": dd,
                          "pem_rss": rss_val, "rss_fit": pem_c.rss,
                          "criterion": float(crit_val),
                          "arx": arx_c, "pem": pem_c})
            if nn not in order_bic or crit_val < order_bic[nn]:
                order_bic[nn] = float(crit_val)

    if not table:
        raise ValueError("structure selection failed: no candidate could be fitted")

    best_val = min(order_bic.values()) if order_bic else None
    # Simplicity tie-break: when several orders are within
    # ``simplicity_window`` criterion units of the best, the evidence for
    # the extra dynamics is weak -- pick the lowest such order (Occam).
    eligible = sorted(n for n, v in order_bic.items()
                      if best_val is not None and v <= best_val + simplicity_window)
    n_pick = eligible[0]
    best = min((r for r in table if r["n"] == n_pick),
               key=lambda r: r["criterion"])
    # refit the chosen structure with the noise-appropriate estimator
    # (LS for process noise, IV for measurement noise) + full covariance,
    # and the PEM on *all* training rows for the final coefficients
    from .arx import fit_structure_auto
    arx_final, _info = fit_structure_auto(
        u, y, ts, best["n"], best["m"], best["d"],
        exclude_rows=exclude_rows, criterion=criterion,
        input_unit=input_unit, output_unit=output_unit)
    pem_final = identify_pem(u, y, ts, train_idx=train_idx,
                             structure=arx_final, input_unit=input_unit,
                             output_unit=output_unit)
    # annotate with stage-1 sweep info for reporting
    arx_final.sweep = [e for st in arx_stage1 for e in st.sweep]
    return StructureSelection(arx=arx_final, pem=pem_final, table=table,
                              order_bic=order_bic, criterion=criterion,
                              simplicity_window=float(simplicity_window))


# --------------------------------------------------------------------------- #
def simulate_oe(a, b, d: int, u: np.ndarray, y0: float = 0.0) -> np.ndarray:
    """Free-run simulation of  y[k] = A y[k-1] + B u[k-1-d].

    Zero initial state goes through ``lfilter`` (fast path); a sequential
    recursion is used when the initial output y0 is non-zero.
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    u = np.asarray(u, dtype=float)
    if abs(y0) < 1e-15:
        b_conv = np.concatenate([np.zeros(int(d) + 1), b])
        a_conv = np.concatenate([[1.0], -a])
        with np.errstate(over="ignore", invalid="ignore"):
            return lfilter(b_conv, a_conv, u)
    n = len(u)
    yh = np.empty(n)
    yh[0] = y0
    for k in range(1, n):
        s = 0.0
        for i, ai in enumerate(a):
            idx = k - 1 - i
            if idx >= 0:
                s += ai * yh[idx]
        for j, bj in enumerate(b):
            idx = k - 1 - int(d) - j
            if idx >= 0:
                s += bj * u[idx]
        yh[k] = s
    return yh


def identify_pem(u: np.ndarray, y: np.ndarray, ts: float,
                 train_idx: np.ndarray | None = None,
                 structure=None, x0: np.ndarray | None = None,
                 input_unit: str = "", output_unit: str = "") -> PemResult:
    """Nonlinear OE fit of (a, b) for a fixed ARX structure + delay.

    ``structure`` is an :class:`~simforge.id.arx.ArxResult` providing
    orders and delay (and the linear-least-squares start point).  When
    omitted, a first-order ARX model is fitted first.
    """
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(u) != len(y):
        raise ValueError("u and y must have equal length")

    if structure is None:
        from .arx import identify_arx_auto
        structure = identify_arx_auto(u, y, ts, max_order=1)

    n = structure.n_order
    m = structure.m_order
    d = int(structure.delay)

    if train_idx is None:
        train_idx = np.arange(len(u))
    train_idx = np.asarray(train_idx, dtype=int)

    if x0 is None:
        x0 = np.concatenate([structure.a_vec, structure.b_vec])
    x0 = np.asarray(x0, dtype=float)

    # bounds: strictly stable for order 1, sane ranges otherwise
    lo = np.concatenate([np.full(n, -0.9999 if n == 1 else -2.0),
                         np.full(m, -1e5)])
    hi = np.concatenate([np.full(n, 0.9999 if n == 1 else 2.0),
                         np.full(m, 1e5)])
    x0 = np.clip(x0, lo + 1e-12, hi - 1e-12)

    def residual(params):
        a_p = params[:n]
        b_p = params[n:]
        yh = simulate_oe(a_p, b_p, d, u, y0=0.0)
        r = yh[train_idx] - y[train_idx]
        if not np.all(np.isfinite(r)):
            return np.full(len(train_idx), 1e9)
        return r

    import warnings

    with warnings.catch_warnings():
        # wild intermediate parameters of unstable candidate structures
        # can overflow inside lfilter / TRF internals -- the residual
        # guard below handles the resulting non-finite values
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(over="ignore", invalid="ignore"):
            sol = least_squares(residual, x0, bounds=(lo, hi), method="trf",
                                x_scale="jac", ftol=1e-12, xtol=1e-12,
                                max_nfev=500)
    a_fit = np.asarray(sol.x[:n])
    b_fit = np.asarray(sol.x[n:])
    rss = float(2.0 * sol.cost)

    tf = TransferFunction(a=a_fit, b=b_fit, delay=d, ts=ts,
                          input_unit=input_unit, output_unit=output_unit)
    if not tf.is_stable:
        raise ValueError("PEM solution is unstable -- consider a different "
                         "structure or better excitation")

    K = float(tf.dc_gain)
    p = n + m
    dof = max(len(train_idx) - p, 1)
    se = np.full(p, np.nan)
    try:
        cov = np.linalg.pinv(sol.jac.T @ sol.jac) * (rss / dof)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except Exception:
        pass

    S = 1.0 - float(np.sum(a_fit))
    if abs(S) > 1e-12:
        g = np.concatenate([np.full(n, K / S), np.full(m, 1.0 / S)])
        se_K = float(np.sqrt(max(float(g @ cov @ g), 0.0))) if np.all(
            np.isfinite(se)) else float("nan")
        K_ci = (max(K - 1.96 * se_K, 0.0), K + 1.96 * se_K)
    else:  # pragma: no cover
        K_ci = (float("nan"), float("nan"))

    tau: float | None = None
    tau_ci: tuple[float, float] | None = None
    if n == 1 and 0.0 < a_fit[0] < 1.0:
        tau = float(first_order_from_arx(a_fit[0], b_fit[0], ts)[1])
        if np.isfinite(se[0]):
            dtau = ts / (a_fit[0] * np.log(a_fit[0]) ** 2)
            se_tau = float(abs(dtau) * se[0])
            tau_ci = (max(tau - 1.96 * se_tau, 1e-6), tau + 1.96 * se_tau)
    else:
        tau = tf.tau_eff()

    return PemResult(K=K, tau=tau, td=d * ts, ts=ts,
                     a_vec=a_fit, b_vec=b_fit, delay=d,
                     K_ci=K_ci, tau_ci=tau_ci, rss=rss,
                     n_rows=len(train_idx), success=bool(sol.success),
                     message=str(sol.message), nfev=int(sol.nfev),
                     input_unit=input_unit, output_unit=output_unit)
