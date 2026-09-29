"""End-to-end identification pipeline: load -> segment -> identify -> validate."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .arx import ArxResult, identify_arx
from .io import MotorData
from .preprocess import (Segment, complement_indices, find_segments,
                         holdout_indices, steady_window)
from .stepfit import PemResult, identify_pem
from .validate import (CheckResult, cross_corr_max, estimate_frf, ljung_box,
                       rmse, steady_state_error_pct, vaf, acceptance_checks)


@dataclass
class IdentificationResult:
    data: MotorData
    segments: list[Segment]
    train_idx: np.ndarray
    hold_idx: np.ndarray
    arx: ArxResult
    pem: PemResult
    v_arx: np.ndarray           # free-run ARX simulation (full record)
    v_pem: np.ndarray           # free-run PEM simulation (full record)
    e_free: np.ndarray          # v_meas - v_arx (full record)
    e_fit: np.ndarray           # one-step innovations on fitted rows
    metrics: dict
    checks: list[CheckResult]
    frf: tuple = field(repr=False, default=())

    @property
    def has_holdout(self) -> bool:
        return len(self.hold_idx) > 0

    @property
    def train_idx_excluded_mask(self) -> np.ndarray:
        mask = np.zeros(self.data.n, dtype=bool)
        mask[self.hold_idx] = True
        return mask


def run_identification(data: MotorData, holdout_duty: float | None = None,
                       max_delay: int | None = None, lags: int = 10,
                       frf_nperseg: int | None = None) -> IdentificationResult:
    """Run the full identification and validation pipeline on one recording."""
    u = data.u
    v = data.y
    ts = data.ts

    segments = find_segments(data)
    if holdout_duty is not None:
        hold_idx = holdout_indices(data, segments, holdout_duty)
        train_idx = complement_indices(data.n, hold_idx)
        exclude_mask = np.zeros(data.n, dtype=bool)
        exclude_mask[hold_idx] = True
    else:
        hold_idx = np.empty(0, dtype=int)
        train_idx = np.arange(data.n)
        exclude_mask = None

    # ---- identification ------------------------------------------------
    arx = identify_arx(u, v, ts, max_delay=max_delay, exclude_rows=exclude_mask)
    pem = identify_pem(u, v, ts, train_idx=train_idx)

    # ---- simulations ---------------------------------------------------
    v_arx = arx.simulate(u)
    v_pem = pem.simulate_velocity(u)
    e_free = v - v_arx

    # one-step innovations on exactly the rows used for the ARX fit
    ks = np.arange(arx.delay, data.n - 1)
    e_full = v[ks + 1] - arx.a * v[ks] - arx.b * u[ks - arx.delay]
    e_fit = e_full[arx.row_idx - arx.delay]

    # ---- metrics -------------------------------------------------------
    m: dict = {}
    m["vaf_train_arx"] = vaf(v[train_idx], v_arx[train_idx])
    m["vaf_train_pem"] = vaf(v[train_idx], v_pem[train_idx])
    m["rmse_train_arx"] = rmse(v[train_idx], v_arx[train_idx])
    m["rmse_train_pem"] = rmse(v[train_idx], v_pem[train_idx])

    if hold_idx.size:
        m["vaf_holdout_arx"] = vaf(v[hold_idx], v_arx[hold_idx])
        m["vaf_holdout_pem"] = vaf(v[hold_idx], v_pem[hold_idx])
        # steady state of the held-out pulse itself (not the rest after it)
        pulse = next((s for s in segments
                      if abs(s.duty - (holdout_duty or 0.0)) <= 1e-6
                      and abs(s.duty) > 1e-9), None)
        if pulse is not None:
            w = steady_window(pulse, settle_frac=0.5)
            m["ss_err_holdout_arx_pct"] = steady_state_error_pct(v[w], v_arx[w])
            m["ss_err_holdout_pem_pct"] = steady_state_error_pct(v[w], v_pem[w])
        else:  # pragma: no cover
            m["ss_err_holdout_arx_pct"] = float("nan")
            m["ss_err_holdout_pem_pct"] = float("nan")

    q, p = ljung_box(e_fit, lags=lags, n_params=2)
    m["ljung_box_Q"] = q
    m["ljung_box_p"] = p
    m["ljung_box_lags"] = float(min(lags, len(e_fit) - 2))
    x_max, x_band, _ = cross_corr_max(e_fit, u[arx.row_idx])
    m["xcorr_max"] = x_max
    m["xcorr_band"] = x_band

    m["arx_K"] = arx.K
    m["arx_tau"] = arx.tau
    m["arx_td"] = arx.td
    m["pem_K"] = pem.K
    m["pem_tau"] = pem.tau
    m["pem_td"] = pem.td
    m["k_diff_pct"] = abs(arx.K - pem.K) / max(abs(pem.K), 1e-9) * 100.0
    m["tau_diff_pct"] = abs(arx.tau - pem.tau) / max(pem.tau, 1e-9) * 100.0
    m["K_ci_rel_pct"] = (arx.K_ci[1] - arx.K_ci[0]) / max(arx.K, 1e-9) * 100.0
    m["tau_ci_rel_pct"] = (arx.tau_ci[1] - arx.tau_ci[0]) / max(arx.tau, 1e-9) * 100.0
    m["bandwidth_hz"] = arx.bandwidth_hz
    m["ts_s"] = ts
    m["fs_hz"] = data.fs

    # optional references / ground truth from metadata
    ref_k = data.meta_number("ref_k")
    if ref_k is not None:
        m["ref_k_pct"] = ref_k
    true_k = data.meta_number("true_k")
    true_tau = data.meta_number("true_tau_s")
    true_td = data.meta_number("true_td_s")
    if true_k and true_tau:
        m["true_k"] = true_k
        m["true_tau_s"] = true_tau
        m["true_td_s"] = true_td if true_td is not None else 0.0
        m["k_err_pct"] = abs(arx.K - true_k) / true_k * 100.0
        m["tau_err_pct"] = abs(arx.tau - true_tau) / true_tau * 100.0
        m["td_err_s"] = abs(arx.td - m["true_td_s"])
        m["k_err_pct_pem"] = abs(pem.K - true_k) / true_k * 100.0
        m["tau_err_pct_pem"] = abs(pem.tau - true_tau) / true_tau * 100.0

    checks = acceptance_checks(m, has_holdout=hold_idx.size > 0)
    frf = estimate_frf(u, v, data.fs, nperseg=frf_nperseg)

    return IdentificationResult(
        data=data, segments=segments, train_idx=train_idx, hold_idx=hold_idx,
        arx=arx, pem=pem, v_arx=v_arx, v_pem=v_pem, e_free=e_free,
        e_fit=e_fit, metrics=m, checks=checks, frf=frf,
    )
