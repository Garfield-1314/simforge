"""End-to-end regression tests on synthetic motor data (known ground truth).

The whole pipeline -- CSV round-trip, segmentation, ARX + PEM
identification, five-layer validation, report generation -- must recover
the true parameters within 2 % and pass every acceptance check.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from simforge.id import check_data, load_motor_csv, run_identification, save_motor_csv
from simforge.id.arx import identify_arx
from simforge.id.io import DataFormatError, SystemData
from simforge.id.preprocess import find_segments, holdout_indices
from simforge.id.report import build_figures, write_report
from simforge.id.simulator import TRUE_K, TRUE_TAU, TRUE_TD, generate_dataset
from simforge.id.stepfit import identify_pem
from simforge.id.validate import cross_corr_max, ljung_box, vaf

FS = 500.0


@pytest.fixture(scope="module")
def sim_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "motor_sim.csv"
    generate_dataset(path, fs=FS, seed=42)
    return path


@pytest.fixture(scope="module")
def sim_data(sim_csv):
    return load_motor_csv(sim_csv)


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #
def test_csv_roundtrip(tmp_path, sim_data):
    out = tmp_path / "roundtrip.csv"
    save_motor_csv(out, sim_data)
    again = load_motor_csv(out)
    assert again.n == sim_data.n
    assert again.fs == pytest.approx(FS)
    assert again.counts_per_rev == 4096
    np.testing.assert_array_equal(again.enc_delta, sim_data.enc_delta)
    np.testing.assert_allclose(again.pwm, sim_data.pwm, atol=1e-3)
    assert float(again.meta["true_k"]) == pytest.approx(TRUE_K)
    assert check_data(again) == []


def test_time_axis_validation():
    with pytest.raises(DataFormatError):
        SystemData(t=[0.0, 0.002, 0.001], u=[0, 0, 0], y=[0, 0, 0],
                   fs=500)


def test_speed_conversion(sim_data):
    # 4096 counts/rev at 500 Hz: 81.92 counts/sample == 600 rpm
    expected_per_sample = 600.0 * 4096 / 60.0 / FS
    assert expected_per_sample == pytest.approx(81.92, rel=1e-6)
    idx = np.flatnonzero(sim_data.speed_rpm > 580)[0]
    assert sim_data.enc_delta[idx] == pytest.approx(82, abs=2)


# --------------------------------------------------------------------------- #
# segmentation
# --------------------------------------------------------------------------- #
def test_segments_and_holdout(sim_data):
    segs = find_segments(sim_data)
    duties = [s.duty for s in segs]
    assert duties == [0.0, 20.0, 0.0, 50.0, 0.0, -50.0, 0.0]
    hold = holdout_indices(sim_data, segs, 50.0)
    # pulse starts at 6.0 s and the region extends through the rest after it
    assert hold[0] == int(6.0 * FS)
    assert hold[-1] == int(11.0 * FS) - 1
    with pytest.raises(ValueError):
        holdout_indices(sim_data, segs, 33.0)


# --------------------------------------------------------------------------- #
# identification accuracy (ground truth recovery)
# --------------------------------------------------------------------------- #
def test_arx_recovers_truth(sim_data):
    r = identify_arx(sim_data.pwm, sim_data.speed_rpm, sim_data.ts)
    assert r.delay == 2                      # true td = 4 ms = 2 samples
    assert abs(r.K - TRUE_K) / TRUE_K < 0.02
    assert abs(r.tau - TRUE_TAU) / TRUE_TAU < 0.02
    assert r.td == pytest.approx(TRUE_TD, abs=sim_data.ts)


def test_pem_recovers_truth(sim_data):
    r = identify_pem(sim_data.pwm, sim_data.speed_rpm, sim_data.ts)
    # PEM fits the noisy speed directly (no regressor), so its variance is
    # somewhat larger than ARX's -- the acceptance limit for the method
    # agreement is 5 %.
    assert abs(r.K - TRUE_K) / TRUE_K < 0.02
    assert abs(r.tau - TRUE_TAU) / TRUE_TAU < 0.05
    assert abs(r.td - TRUE_TD) < max(sim_data.ts, 0.1 * TRUE_TD)


def test_model_matches_simulation(sim_data):
    r = identify_pem(sim_data.pwm, sim_data.speed_rpm, sim_data.ts)
    y = r.simulate_velocity(sim_data.pwm)
    assert vaf(sim_data.speed_rpm, y) > 98.0


# --------------------------------------------------------------------------- #
# validation layer
# --------------------------------------------------------------------------- #
def test_ljung_box_detects_structure():
    rng = np.random.default_rng(0)
    white = rng.normal(size=4000)
    q, p = ljung_box(white, lags=10)
    assert p > 0.05
    colored = np.convolve(white, np.ones(20) / 20, mode="same")  # moving average
    q2, p2 = ljung_box(colored, lags=10)
    assert p2 < 1e-6


def test_cross_corr_band():
    rng = np.random.default_rng(1)
    e = rng.normal(size=4000)
    x = rng.normal(size=4000)
    m, band, _ = cross_corr_max(e, x, nlags=10)
    assert m < band
    m2, _, _ = cross_corr_max(e, np.roll(e, 3), nlags=10)
    assert m2 > band


def test_pipeline_all_checks_pass(sim_data):
    res = run_identification(sim_data, holdout_duty=50.0)
    failed = [c for c in res.checks if c.status == "fail"]
    assert not failed, f"failed checks: {[(c.name, c.value) for c in failed]}"
    m = res.metrics
    assert m["vaf_train_arx"] > 95.0
    assert m["vaf_holdout_arx"] > 90.0
    assert m["ss_err_holdout_arx_pct"] < 3.0
    assert m["ljung_box_p"] > 0.05
    assert m["k_err_pct"] < 2.0
    assert m["tau_err_pct"] < 2.0


def test_pipeline_without_holdout(sim_data):
    res = run_identification(sim_data, holdout_duty=None)
    statuses = {c.name.split()[0]: c.status for c in res.checks}
    assert statuses["2"] == "skip"
    assert not [c for c in res.checks if c.status == "fail"]


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def test_report_outputs(tmp_path, sim_data):
    res = run_identification(sim_data, holdout_duty=50.0)
    out = tmp_path / "report"
    paths = write_report(res, out)

    assert (out / "report.html").stat().st_size > 1_000_000  # embeds plotly.js
    assert (out / "report.md").stat().st_size > 1000
    model = json.loads((out / "model.json").read_text(encoding="utf-8"))
    assert model["model_type"] == "first_order_with_delay"
    assert abs(model["arx"]["K_rpm_per_pct"] - TRUE_K) / TRUE_K < 0.02
    assert model["operating_conditions"]["load"] == "wheel_suspended"
    assert len(paths["pngs"]) == 8
    for p in paths["pngs"]:
        import pathlib
        assert pathlib.Path(p).stat().st_size > 5000

    figs = build_figures(res)
    assert [f["name"] for f in figs] == [
        "T1_overview", "T2_step_fit", "T3_holdout", "T4_residual",
        "T5_methods", "F1_bode_mag", "F2_bode_phase", "F3_residual_psd"]
