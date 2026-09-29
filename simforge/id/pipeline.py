"""End-to-end identification pipeline: load -> segment -> identify -> validate."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .arx import ArxResult, identify_arx_auto
from .io import SystemData
from .preprocess import (Segment, complement_indices, find_segments,
                         holdout_indices, steady_window)
from .stepfit import PemResult, select_structure_pem
from .validate import (CheckResult, Thresholds, acceptance_checks,
                       cross_corr_max, estimate_frf, ljung_box, rmse,
                       steady_state_error_pct, vaf)


@dataclass
class IdentificationResult:
    data: SystemData
    segments: list[Segment]
    train_idx: np.ndarray
    hold_idx: np.ndarray
    arx: ArxResult
    pem: PemResult
    v_arx: np.ndarray           # free-run ARX simulation (full record)
    v_pem: np.ndarray           # free-run PEM simulation (full record)
    e_free: np.ndarray          # y_meas - y_arx (full record)
    e_fit: np.ndarray           # one-step innovations on fitted rows
    metrics: dict
    checks: list[CheckResult]
    thresholds: Thresholds
    frf: tuple = field(repr=False, default=())

    @property
    def has_holdout(self) -> bool:
        return len(self.hold_idx) > 0


def run_identification(data: SystemData, holdout_duty: float | None = None,
                       max_delay: int | None = None, lags: int = 10,
                       frf_nperseg: int | None = None,
                       max_order: int = 1, criterion: str = "bic",
                       thresholds: Thresholds | None = None) -> IdentificationResult:
    """Run the full identification and validation pipeline on one recording.

    ``max_order`` = 1 (default) fits the classic first-order+delay model;
    larger values enable automatic order selection up to that order.
    """
    th = thresholds or Thresholds()
    u = data.u
    y = data.y
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

    # ---- identification (two-stage) --------------------------------------- #
    # stage 1: ARX grid search with BOTH estimators, so the candidate
    # delays cover the LS view (efficient under process noise) and the
    # IV view (consistent under measurement noise)
    stage1s = [identify_arx_auto(u, y, ts, max_order=max_order,
                                 max_delay=max_delay, criterion=criterion,
                                 method=meth, exclude_rows=exclude_mask,
                                 input_unit=data.input_unit,
                                 output_unit=data.output_unit)
               for meth in ("ls", "iv")]
    # stage 2: consistent output-error criterion picks (n, m, d), then the
    # chosen structure gets an auto refit (covariance) + OE-refined PEM fit
    sel = select_structure_pem(u, y, ts, stage1s, train_idx=train_idx,
                               exclude_rows=exclude_mask, criterion=criterion,
                               input_unit=data.input_unit,
                               output_unit=data.output_unit)
    arx, pem = sel.arx, sel.pem

    # ---- simulations ---------------------------------------------------
    v_arx = arx.simulate(u)
    v_pem = pem.simulate_velocity(u)
    e_free = y - v_arx
    e_fit = arx.innovations(y, u)

    # ---- metrics -------------------------------------------------------
    m: dict = {}
    m["order_n"] = sel.n
    m["order_m"] = sel.m
    m["criterion"] = sel.criterion
    m["criterion_value"] = sel.best_value
    m["order_gap"] = sel.order_gap
    m["order_bic"] = {str(k): v for k, v in sel.order_bic.items()}
    m["order_tolerance"] = sel.simplicity_window
    m["delay_samples"] = sel.delay
    m["arx_method"] = arx.method

    m["vaf_train_arx"] = vaf(y[train_idx], v_arx[train_idx])
    m["vaf_train_pem"] = vaf(y[train_idx], v_pem[train_idx])
    m["rmse_train_arx"] = rmse(y[train_idx], v_arx[train_idx])
    m["rmse_train_pem"] = rmse(y[train_idx], v_pem[train_idx])

    if hold_idx.size:
        m["vaf_holdout_arx"] = vaf(y[hold_idx], v_arx[hold_idx])
        m["vaf_holdout_pem"] = vaf(y[hold_idx], v_pem[hold_idx])
        pulse = next((s for s in segments
                      if abs(s.duty - (holdout_duty or 0.0)) <= 1e-6
                      and abs(s.duty) > 1e-9), None)
        if pulse is not None:
            w = steady_window(pulse, settle_frac=0.5)
            m["ss_err_holdout_arx_pct"] = steady_state_error_pct(y[w], v_arx[w])
            m["ss_err_holdout_pem_pct"] = steady_state_error_pct(y[w], v_pem[w])
        else:  # pragma: no cover
            m["ss_err_holdout_arx_pct"] = float("nan")
            m["ss_err_holdout_pem_pct"] = float("nan")

    q, p = ljung_box(e_fit, lags=lags, n_params=arx.n_order + arx.m_order,
                     min_lag=arx.n_order)
    m["ljung_box_Q"] = q
    m["ljung_box_p"] = p
    m["ljung_box_lags"] = float(min(lags, len(e_fit) - 2))
    m["ljung_box_min_lag"] = float(arx.n_order)
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
    if arx.tau is not None and pem.tau is not None and min(arx.tau, pem.tau) > 0:
        m["tau_diff_pct"] = abs(arx.tau - pem.tau) / pem.tau * 100.0
    else:
        m["tau_diff_pct"] = float("nan")
    m["K_ci_rel_pct"] = (arx.K_ci[1] - arx.K_ci[0]) / max(abs(arx.K), 1e-9) * 100.0
    m["pem_K_ci_rel_pct"] = (pem.K_ci[1] - pem.K_ci[0]) / max(abs(pem.K), 1e-9) * 100.0
    if arx.tau_ci is not None:
        m["tau_ci_rel_pct"] = (arx.tau_ci[1] - arx.tau_ci[0]) / max(arx.tau, 1e-9) * 100.0
    else:
        m["tau_ci_rel_pct"] = float("nan")
    if pem.tau_ci is not None and pem.tau is not None:
        m["pem_tau_ci_rel_pct"] = (pem.tau_ci[1] - pem.tau_ci[0]) / max(pem.tau, 1e-9) * 100.0
    else:
        m["pem_tau_ci_rel_pct"] = float("nan")
    m["bandwidth_hz"] = arx.bandwidth_hz
    m["ts_s"] = ts
    m["fs_hz"] = data.fs

    # optional references / ground truth from metadata
    ref_gain = data.meta_number("ref_gain", data.meta_number("ref_k"))
    if ref_gain is not None:
        m["ref_gain"] = ref_gain
    true_k = data.meta_number("true_k")
    true_tau = data.meta_number("true_tau_s")
    true_td = data.meta_number("true_td_s")
    if true_k and true_tau:
        m["true_k"] = true_k
        m["true_tau_s"] = true_tau
        m["true_td_s"] = true_td if true_td is not None else 0.0
        m["k_err_pct"] = abs(arx.K - true_k) / true_k * 100.0
        m["k_err_pct_pem"] = abs(pem.K - true_k) / true_k * 100.0
        if arx.tau is not None:
            m["tau_err_pct"] = abs(arx.tau - true_tau) / true_tau * 100.0
        if pem.tau is not None:
            m["tau_err_pct_pem"] = abs(pem.tau - true_tau) / true_tau * 100.0
        m["td_err_s"] = abs(arx.td - m["true_td_s"])

    checks = acceptance_checks(m, has_holdout=hold_idx.size > 0, th=th)
    frf = estimate_frf(u, y, data.fs, nperseg=frf_nperseg)

    return IdentificationResult(
        data=data, segments=segments, train_idx=train_idx, hold_idx=hold_idx,
        arx=arx, pem=pem, v_arx=v_arx, v_pem=v_pem, e_free=e_free,
        e_fit=e_fit, metrics=m, checks=checks, thresholds=th, frf=frf,
    )
