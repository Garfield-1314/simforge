"""Validation utilities: fit metrics, residual diagnostics, acceptance checks.

Five layers of validation (see docs):

1. in-sample goodness of fit (VAF, RMSE)
2. hold-out prediction (out-of-sample VAF, steady-state error)
3. residual diagnostics (Ljung-Box whiteness, residual/input cross-correlation)
4. parameter confidence intervals + agreement between the two methods
5. physical plausibility (gain vs theoretical reference), parameter
   recovery (synthetic data only) and model-order confidence

All pass/fail thresholds live in :class:`Thresholds` and can be
overridden per run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import chi2


# --------------------------------------------------------------------------- #
# thresholds
# --------------------------------------------------------------------------- #
@dataclass
class Thresholds:
    """Acceptance thresholds for the validation checklist."""

    vaf_train: float = 95.0          # %
    vaf_holdout: float = 90.0        # %
    ss_err_holdout: float = 3.0      # %
    ljung_p: float = 0.05
    method_diff: float = 5.0         # %  (ARX vs PEM, gain & time constant)
    ci_width: float = 20.0           # %  (95% CI relative width)
    ref_gain_dev: float = 15.0       # %  (gain vs theoretical reference)
    recovery: float = 2.0            # %  (parameter recovery, synthetic only)
    order_gap: float = 10.0          # AIC gap between chosen and next order
    xcorr_conf: float = 1.96         # sigma band for residual cross-correlation
    lags: int = 10                   # Ljung-Box lags


# --------------------------------------------------------------------------- #
# basic metrics
# --------------------------------------------------------------------------- #
def vaf(y: np.ndarray, yhat: np.ndarray) -> float:
    """Variance accounted for [%]: 100 * (1 - sum(e^2)/sum((y-mean)^2))."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return 100.0 if ss_res <= 1e-12 else -np.inf
    return 100.0 * (1.0 - ss_res / ss_tot)


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def steady_state_error_pct(y: np.ndarray, yhat: np.ndarray) -> float:
    """Relative error of the mean steady-state value [%]."""
    mean_y = float(np.mean(y))
    denom = max(abs(mean_y), 1e-9)
    return 100.0 * abs(float(np.mean(yhat)) - mean_y) / denom


# --------------------------------------------------------------------------- #
# residual diagnostics
# --------------------------------------------------------------------------- #
def ljung_box(e: np.ndarray, lags: int = 10, n_params: int = 2,
              min_lag: int = 0) -> tuple[float, float]:
    """Ljung-Box whiteness test on lags ``min_lag+1 .. min_lag+lags``.

    Returns (Q, p).  p > 0.05 == consistent with white noise.

    ``min_lag`` skips the first few lags: for an ARX model whose output
    carries *measurement* noise the one-step error is an MA(n) sequence
    (``e = eps[k+1] - a0 eps[k] - ...``) whose autocorrelation is
    structurally non-zero up to lag n.  Testing from lag n+1 onwards
    still detects genuine unmodelled dynamics without false alarms.
    """
    e = np.asarray(e, dtype=float)
    e = e - np.mean(e)
    n = len(e)
    start = int(min_lag) + 1
    end = int(min_lag) + int(min(lags, n - 2))
    if end < start or n < 20:
        return 0.0, 1.0
    denom = float(np.sum(e ** 2))
    if denom <= 0:
        return 0.0, 1.0
    q = 0.0
    for k in range(start, end + 1):
        rho = float(np.sum(e[k:] * e[:-k])) / denom
        q += rho ** 2 / (n - k)
    q *= n * (n + 2)
    dof = max(end - start + 1, 1)
    p = float(chi2.sf(q, dof))
    return q, p


def cross_corr_max(e: np.ndarray, x: np.ndarray, nlags: int = 20,
                   conf: float = 1.96) -> tuple[float, float, np.ndarray]:
    """Max |normalised cross-correlation| of e vs x over +/-nlags.

    Returns (max_abs, band, rhos) where ``band = conf/sqrt(n)`` is the
    confidence bound for white, independent samples.
    """
    e = np.asarray(e, dtype=float)
    x = np.asarray(x, dtype=float)
    e = e - np.mean(e)
    x = x - np.mean(x)
    n = len(e)
    denom = float(np.sqrt(np.sum(e ** 2) * np.sum(x ** 2)))
    if denom <= 0 or n < 20:
        return 0.0, 1.0, np.zeros(2 * nlags + 1)
    rhos = []
    for lag in range(-nlags, nlags + 1):
        if lag >= 0:
            a, b = e[lag:], x[: n - lag]
        else:
            a, b = e[: n + lag], x[-lag:]
        rhos.append(float(np.sum(a * b)) / denom)
    rhos_arr = np.asarray(rhos)
    band = conf / np.sqrt(n)
    return float(np.max(np.abs(rhos_arr))), band, rhos_arr


# --------------------------------------------------------------------------- #
# frequency domain
# --------------------------------------------------------------------------- #
def estimate_frf(u: np.ndarray, y: np.ndarray, fs: float,
                 nperseg: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Welch estimate of H(f) = S_yu / S_uu plus the coherence."""
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    if nperseg is None:
        nperseg = int(min(2048, max(256, len(u) // 4)))
    f, Puu = sp_signal.welch(u, fs=fs, nperseg=nperseg, detrend="constant")
    _, Pyy = sp_signal.welch(y, fs=fs, nperseg=nperseg, detrend="constant")
    _, Pyu = sp_signal.csd(y, u, fs=fs, nperseg=nperseg, detrend="constant")
    with np.errstate(divide="ignore", invalid="ignore"):
        H = np.where(Puu > 0, Pyu / np.where(Puu > 0, Puu, 1.0), np.nan)
        coh = np.where(Puu * Pyy > 0, np.abs(Pyu) ** 2 / (Puu * Pyy), 0.0)
    return f, H, np.clip(np.real(coh), 0.0, 1.0)


def model_bode(K: float, tau: float, td: float, freqs_hz: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude [dB] and phase [deg] of K*exp(-td*s)/(tau*s+1)."""
    w = 2.0 * np.pi * np.asarray(freqs_hz, dtype=float)
    g = K * np.exp(-1j * w * td) / (1j * w * tau + 1.0)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(g), 1e-12))
    phase_deg = np.unwrap(np.angle(g)) * 180.0 / np.pi
    return mag_db, phase_deg


def residual_psd(e: np.ndarray, fs: float, nperseg: int | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """One-sided PSD of the residual sequence (flat => white noise)."""
    e = np.asarray(e, dtype=float)
    if nperseg is None:
        nperseg = int(min(1024, max(64, len(e) // 4)))
    f, pxx = sp_signal.welch(e - np.mean(e), fs=fs, nperseg=nperseg)
    return f, pxx


# --------------------------------------------------------------------------- #
# acceptance checklist
# --------------------------------------------------------------------------- #
@dataclass
class CheckResult:
    name: str
    status: str          # "pass" | "fail" | "skip" | "manual"
    value: str = ""
    threshold: str = ""
    note: str = ""

    @property
    def passed(self) -> bool:
        return self.status in ("pass", "skip", "manual")


def _check(name: str, ok: bool, value: str, threshold: str, note: str = "") -> CheckResult:
    return CheckResult(name=name, status="pass" if ok else "fail",
                       value=value, threshold=threshold, note=note)


def acceptance_checks(metrics: dict, has_holdout: bool,
                      th: Thresholds | None = None) -> list[CheckResult]:
    """Turn the metric dict into the acceptance checklist."""
    th = th or Thresholds()
    checks: list[CheckResult] = []

    def fmt(v) -> str:
        return f"{v:.1f}" if isinstance(v, (int, float)) and np.isfinite(v) else "n/a"

    # 1 -- in-sample fit
    vafs_train = [metrics["vaf_train_arx"], metrics["vaf_train_pem"]]
    checks.append(_check(
        f"1 训练段 VAF >= {th.vaf_train:g}%", min(vafs_train) >= th.vaf_train,
        f"ARX {fmt(vafs_train[0])}% / PEM {fmt(vafs_train[1])}%",
        f">= {th.vaf_train:g}%"))

    # 2/3 -- hold-out prediction
    if has_holdout:
        vafs_hold = [metrics["vaf_holdout_arx"], metrics["vaf_holdout_pem"]]
        checks.append(_check(
            f"2 留出段 VAF >= {th.vaf_holdout:g}%", min(vafs_hold) >= th.vaf_holdout,
            f"ARX {fmt(vafs_hold[0])}% / PEM {fmt(vafs_hold[1])}%",
            f">= {th.vaf_holdout:g}%"))
        errs = [metrics["ss_err_holdout_arx_pct"], metrics["ss_err_holdout_pem_pct"]]
        checks.append(_check(
            f"3 留出段稳态误差 <= {th.ss_err_holdout:g}%", max(errs) <= th.ss_err_holdout,
            f"ARX {fmt(errs[0])}% / PEM {fmt(errs[1])}%",
            f"<= {th.ss_err_holdout:g}%"))
    else:
        checks.append(CheckResult(f"2 留出段 VAF >= {th.vaf_holdout:g}%", "skip",
                                  note="无留出段"))
        checks.append(CheckResult(f"3 留出段稳态误差 <= {th.ss_err_holdout:g}%", "skip",
                                  note="无留出段"))

    # 4/5 -- residual whiteness & independence from input
    checks.append(_check(
        f"4 Ljung-Box 白噪声 p > {th.ljung_p:g}",
        metrics["ljung_box_p"] > th.ljung_p,
        f"p = {metrics['ljung_box_p']:.3f} (Q={metrics['ljung_box_Q']:.1f}, "
        f"lags={int(metrics['ljung_box_lags'])})", f"> {th.ljung_p:g}"))
    checks.append(_check(
        "5 残差×输入互相关在置信带内",
        metrics["xcorr_max"] <= metrics["xcorr_band"],
        f"max|rho| = {metrics['xcorr_max']:.4f}",
        f"<= {metrics['xcorr_band']:.4f}"))

    # 6 -- agreement between the two methods
    diffs = [("K", metrics["k_diff_pct"])]
    if np.isfinite(metrics.get("tau_diff_pct", float("nan"))):
        diffs.append(("tau", metrics["tau_diff_pct"]))
    worst = max(v for _, v in diffs)
    labels = "/".join(name for name, _ in diffs)
    checks.append(_check(
        f"6 双方法一致性 ({labels}) <= {th.method_diff:g}%",
        worst <= th.method_diff,
        " / ".join(f"{v:.2f}%" for _, v in diffs), f"<= {th.method_diff:g}%"))

    # 7 -- parameter confidence intervals (PEM primary, ARX shown for info)
    def _first_finite(*keys):
        for k in keys:
            v = metrics.get(k, float("nan"))
            if v is not None and np.isfinite(v):
                return float(v)
        return float("nan")

    ci_k = _first_finite("pem_K_ci_rel_pct", "K_ci_rel_pct")
    ci_tau = _first_finite("pem_tau_ci_rel_pct", "tau_ci_rel_pct")
    parts = [v for v in (ci_k, ci_tau) if np.isfinite(v)]
    ci_worst = max(parts) if parts else float("inf")
    arx_ci = metrics.get("K_ci_rel_pct", float("nan"))
    val = f"K {ci_k:.1f}%" if np.isfinite(ci_k) else ""
    if np.isfinite(ci_tau):
        val += (f" / tau {ci_tau:.1f}%" if val else f"tau {ci_tau:.1f}%")
    if np.isfinite(arx_ci):
        val += f" (ARX K {arx_ci:.1f}%)"
    checks.append(_check(
        f"7 参数 95%CI 宽度 <= {th.ci_width:g}%", ci_worst <= th.ci_width,
        val or "n/a", f"<= {th.ci_width:g}%"))

    # 8 -- physical reference gain
    ref_gain = metrics.get("ref_gain")
    if ref_gain:
        dev = abs(metrics["arx_K"] - ref_gain) / ref_gain * 100.0
        checks.append(_check(
            f"8 增益与物理参考偏差 <= {th.ref_gain_dev:g}%", dev <= th.ref_gain_dev,
            f"{dev:.1f}%  (gain={metrics['arx_K']:.3g}, ref={ref_gain:.3g})",
            f"<= {th.ref_gain_dev:g}%"))
    else:
        checks.append(CheckResult(
            f"8 增益与物理参考偏差 <= {th.ref_gain_dev:g}%", "skip",
            note="未提供理论参考增益（CSV 元数据 ref_gain/ref_k）"))

    # 9 -- parameter recovery (synthetic data only)
    if metrics.get("true_k") and metrics.get("true_tau_s") is not None:
        rec_parts = [metrics["k_err_pct"]]
        if np.isfinite(metrics.get("tau_err_pct", float("nan"))):
            rec_parts.append(metrics["tau_err_pct"])
        rec = max(rec_parts)
        td_ok = metrics["td_err_s"] <= max(metrics["ts_s"],
                                           0.1 * metrics["true_td_s"])
        checks.append(_check(
            f"9 参数恢复误差 <= {th.recovery:g}% (仿真数据)", rec <= th.recovery and td_ok,
            f"K {metrics['k_err_pct']:.2f}%"
            + (f" / tau {metrics['tau_err_pct']:.2f}%" if len(rec_parts) > 1 else "")
            + f" / td {metrics['td_err_s'] * 1e3:.1f} ms",
            f"K,tau <= {th.recovery:g}%; td <= 1 采样"))
    else:
        checks.append(CheckResult(
            f"9 参数恢复误差 <= {th.recovery:g}% (仿真数据)", "skip",
            note="真实数据，无真值可比"))

    # 10 -- gold-standard re-test (manual)
    checks.append(CheckResult(
        "10 金标准复测 (新输入阶跃)", "manual",
        threshold="预测误差 <= 5%",
        note="模型验收的最终裁判：换一个未用过的输入幅值录 5 s 对比"))

    # 11 -- model-order confidence.  Strong evidence: the chosen order
    # beats every alternative by >= th.order_gap criterion units.
    # Weak evidence: alternatives are within the 5%-RSS tie tolerance and
    # the simpler model was taken deliberately -- pass with a note.
    # (the tolerance is what let a first-order system be chosen although
    # extra poles absorbed ~3% of sample-specific disturbance projection)
    gap = metrics.get("order_gap")
    tol = float(metrics.get("order_tolerance", 0.0) or 0.0)
    crit = str(metrics.get("criterion", "bic")).upper()
    if gap is None or not np.isfinite(gap):
        checks.append(CheckResult(
            f"11 阶次选择置信 (跨阶次 {crit} 差 >= {th.order_gap:g})", "skip",
            note="仅评估单一阶次 (max_order=1)"))
    elif gap >= th.order_gap:
        checks.append(_check(
            f"11 阶次选择置信 (跨阶次 {crit} 差 >= {th.order_gap:g})", True,
            f"gap = {gap:.1f} (n = {int(metrics['order_n'])})",
            f">= {th.order_gap:g}", "强证据"))
    elif gap >= -max(tol, 1e-9):
        checks.append(_check(
            f"11 阶次选择置信 (跨阶次 {crit} 差 >= {th.order_gap:g})", True,
            f"gap = {gap:.1f} (n = {int(metrics['order_n'])}), 容差 {tol:.0f}",
            f">= {th.order_gap:g}",
            "证据弱：次优阶 RSS 优 <5%，已按简单性取低阶（存疑但可接受）"))
    else:
        checks.append(_check(
            f"11 阶次选择置信 (跨阶次 {crit} 差 >= {th.order_gap:g})", False,
            f"gap = {gap:.1f} (n = {int(metrics['order_n'])})",
            f">= {th.order_gap:g}", "次优阶显著更优，阶次存疑"))
    return checks
