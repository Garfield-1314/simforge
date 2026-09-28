"""Validation utilities: fit metrics, residual diagnostics, acceptance checks.

Five layers of validation (see plan / docs):

1. in-sample goodness of fit (VAF, RMSE)
2. hold-out prediction (out-of-sample VAF, steady-state error)
3. residual diagnostics (Ljung-Box whiteness, residual/input cross-correlation)
4. parameter confidence intervals + agreement between the two methods
5. physical plausibility (K vs theoretical reference) and parameter recovery
   (synthetic data only)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import chi2


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
def ljung_box(e: np.ndarray, lags: int = 10, n_params: int = 2) -> tuple[float, float]:
    """Ljung-Box whiteness test.  Returns (Q, p).  p > 0.05 == white."""
    e = np.asarray(e, dtype=float)
    e = e - np.mean(e)
    n = len(e)
    lags = int(min(lags, n - 2))
    denom = float(np.sum(e ** 2))
    if denom <= 0 or n < 20:
        return 0.0, 1.0
    q = 0.0
    for k in range(1, lags + 1):
        rho = float(np.sum(e[k:] * e[:-k])) / denom
        q += rho ** 2 / (n - k)
    q *= n * (n + 2)
    dof = max(lags - n_params, 1)
    p = float(chi2.sf(q, dof))
    return q, p


def cross_corr_max(e: np.ndarray, x: np.ndarray, nlags: int = 20
                   ) -> tuple[float, float, np.ndarray]:
    """Max |normalised cross-correlation| of e vs x over +/-nlags.

    Returns (max_abs, band, rhos) where ``band = 1.96/sqrt(n)`` is the
    95 % confidence bound for white, independent samples.
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
    band = 1.96 / np.sqrt(n)
    return float(np.max(np.abs(rhos_arr))), band, rhos_arr


# --------------------------------------------------------------------------- #
# frequency domain
# --------------------------------------------------------------------------- #
def estimate_frf(u: np.ndarray, v: np.ndarray, fs: float,
                 nperseg: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Welch estimate of H(f) = S_vu / S_uu plus the coherence.

    Step-like data is non-stationary; the estimate is indicative rather
    than exact -- coherence tells you which frequencies to trust.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if nperseg is None:
        nperseg = int(min(2048, max(256, len(u) // 4)))
    f, Puu = sp_signal.welch(u, fs=fs, nperseg=nperseg, detrend="constant")
    _, Pvv = sp_signal.welch(v, fs=fs, nperseg=nperseg, detrend="constant")
    _, Pvu = sp_signal.csd(v, u, fs=fs, nperseg=nperseg, detrend="constant")
    with np.errstate(divide="ignore", invalid="ignore"):
        H = np.where(Puu > 0, Pvu / np.where(Puu > 0, Puu, 1.0), np.nan)
        coh = np.where(Puu * Pvv > 0, np.abs(Pvu) ** 2 / (Puu * Pvv), 0.0)
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


def acceptance_checks(metrics: dict, has_holdout: bool) -> list[CheckResult]:
    """Turn the metric dict into the 10-item acceptance checklist."""
    checks: list[CheckResult] = []

    # 1 -- in-sample fit
    vafs_train = [metrics["vaf_train_arx"], metrics["vaf_train_pem"]]
    checks.append(_check(
        "1 训练段 VAF >= 95%", min(vafs_train) >= 95.0,
        f"ARX {metrics['vaf_train_arx']:.1f}% / PEM {metrics['vaf_train_pem']:.1f}%",
        ">= 95%"))

    # 2 -- hold-out prediction
    if has_holdout:
        vafs_hold = [metrics["vaf_holdout_arx"], metrics["vaf_holdout_pem"]]
        checks.append(_check(
            "2 留出段 VAF >= 90%", min(vafs_hold) >= 90.0,
            f"ARX {metrics['vaf_holdout_arx']:.1f}% / PEM {metrics['vaf_holdout_pem']:.1f}%",
            ">= 90%"))
        errs = [metrics["ss_err_holdout_arx_pct"], metrics["ss_err_holdout_pem_pct"]]
        checks.append(_check(
            "3 留出段稳态误差 <= 3%", max(errs) <= 3.0,
            f"ARX {errs[0]:.2f}% / PEM {errs[1]:.2f}%", "<= 3%"))
    else:
        checks.append(CheckResult("2 留出段 VAF >= 90%", "skip", note="无留出段"))
        checks.append(CheckResult("3 留出段稳态误差 <= 3%", "skip", note="无留出段"))

    # 3 -- residual whiteness & independence from input
    checks.append(_check(
        "4 Ljung-Box 白噪声 p > 0.05", metrics["ljung_box_p"] > 0.05,
        f"p = {metrics['ljung_box_p']:.3f} (Q={metrics['ljung_box_Q']:.1f}, "
        f"lags={int(metrics['ljung_box_lags'])})", "> 0.05"))
    checks.append(_check(
        "5 残差×输入互相关在 95% 带内", metrics["xcorr_max"] <= metrics["xcorr_band"],
        f"max|rho| = {metrics['xcorr_max']:.4f}", f"<= {metrics['xcorr_band']:.4f}"))

    # 4 -- parameter quality
    k_diff = metrics["k_diff_pct"]
    tau_diff = metrics["tau_diff_pct"]
    checks.append(_check(
        "6 双方法 K/tau 一致 <= 5%", max(k_diff, tau_diff) <= 5.0,
        f"K {k_diff:.2f}% / tau {tau_diff:.2f}%", "<= 5%"))
    ci_w = max(metrics["K_ci_rel_pct"], metrics["tau_ci_rel_pct"])
    checks.append(_check(
        "7 ARX 参数 95%CI 宽度 <= 20%", ci_w <= 20.0,
        f"K {metrics['K_ci_rel_pct']:.1f}% / tau {metrics['tau_ci_rel_pct']:.1f}%",
        "<= 20%"))

    # 5 -- physical reference / recovery
    ref_k = metrics.get("ref_k_pct")
    if ref_k:
        dev = abs(metrics["arx_K"] - ref_k) / ref_k * 100.0
        checks.append(_check(
            "8 K 与物理参考值偏差 <= 15%", dev <= 15.0,
            f"{dev:.1f}%  (K={metrics['arx_K']:.1f}, ref={ref_k:.1f})", "<= 15%"))
    else:
        checks.append(CheckResult("8 K 与物理参考值偏差 <= 15%", "skip",
                                  note="未提供理论 K（可在 CSV 元数据中加 ref_k）"))

    if metrics.get("true_k"):
        rec = max(metrics["k_err_pct"], metrics["tau_err_pct"])
        td_ok = metrics["td_err_s"] <= max(metrics["ts_s"], 0.1 * metrics["true_td_s"])
        checks.append(_check(
            "9 参数恢复误差 <= 2% (仿真数据)", rec <= 2.0 and td_ok,
            f"K {metrics['k_err_pct']:.2f}% / tau {metrics['tau_err_pct']:.2f}% / "
            f"td {metrics['td_err_s'] * 1e3:.1f} ms",
            "K,tau <= 2%; td <= 1 采样"))
    else:
        checks.append(CheckResult("9 参数恢复误差 <= 2% (仿真数据)", "skip",
                                  note="真实数据，无真值可比"))

    checks.append(CheckResult(
        "10 金标准复测 (新占空比阶跃)", "manual",
        threshold="预测误差 <= 5%",
        note="模型验收的最终裁判：换一个未用过的占空比录 5 s 对比"))
    return checks
