"""Generic-system regression tests (no motor-specific assumptions).

Covers the generalization plan acceptance criteria:
- second-order synthetic system: order 2 selected, wn/zeta/gain/delay
  recovered within 5 %, all acceptance checks pass
- first-order synthetic system: auto-order stays at 1
- generic ``u``/``y`` CSV round-trip with unit metadata
- Thresholds are actually configurable
- transfer-function outputs (characteristic equations, freq response)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from simforge.id import Thresholds, load_system_csv, run_identification, save_system_csv
from simforge.id.excitation import random_dwell
from simforge.id.io import SystemData
from simforge.id.simulator import (generate_second_order_dataset,
                                   second_order_system, simulate_lti)
from simforge.id.tf import TransferFunction

REPO = Path(__file__).resolve().parents[2]
TS = 0.002
FN, ZETA, GAIN, DELAY = 2.0, 0.4, 2.0, 3


@pytest.fixture(scope="module")
def second_order():
    truth_fn = {"wn": 2 * np.pi * FN, "zeta": ZETA, "gain": GAIN, "fn_hz": FN}
    u = random_dwell(8000, -10.0, 10.0, 50, seed=7)
    ad, bd, cd, dd, _ = second_order_system(FN, ZETA, GAIN, TS)
    y = simulate_lti(u, ad, bd, cd, dd, delay=DELAY, meas_noise=0.02, seed=7)
    data = SystemData(t=np.arange(8000) * TS, u=u, y=y, fs=1.0 / TS,
                      input_unit="V", output_unit="rpm", meta={})
    return data, truth_fn


# --------------------------------------------------------------------------- #
def test_second_order_full_recovery(second_order):
    data, truth = second_order
    res = run_identification(data, max_order=3)
    m = res.metrics
    arx = res.arx
    tf = arx.to_tf()

    # structure
    assert m["order_n"] == 2, f"order selected {m['order_n']} (truth 2)"
    assert m["delay_samples"] == DELAY

    # dynamics within 5 %
    ps = tf.poles_s()
    wn_est = abs(ps[0])
    zeta_est = -ps[0].real / abs(ps[0])
    assert abs(wn_est - truth["wn"]) / truth["wn"] < 0.05
    assert abs(zeta_est - truth["zeta"]) / truth["zeta"] < 0.05
    assert abs(tf.dc_gain - GAIN) / GAIN < 0.05

    # all checks pass / skip / manual (no fails)
    failed = [c for c in res.checks if c.status == "fail"]
    assert not failed, [(c.name, c.value) for c in failed]
    assert m["ljung_box_p"] > 0.05
    assert m["vaf_train_arx"] > 95.0
    assert m["k_diff_pct"] <= 5.0
    # estimator should be IV (measurement noise present)
    assert m["arx_method"] == "iv"


def test_first_order_stays_order_one():
    """Auto-order must not over-select on a first-order system."""
    rng = np.random.default_rng(3)
    u = random_dwell(6000, -10.0, 10.0, 80, seed=3)
    a = float(np.exp(-TS / 0.04))
    b = 30.0 * (1.0 - a)
    y = np.empty(6000)
    y[0] = 0.0
    for k in range(1, 6000):
        y[k] = a * y[k - 1] + b * u[k - 1] + rng.normal(0.0, 3.0)
    data = SystemData(t=np.arange(6000) * TS, u=u, y=y, fs=1.0 / TS,
                      input_unit="pct", output_unit="rpm", meta={})
    res = run_identification(data, max_order=3)
    assert res.metrics["order_n"] == 1
    assert not [c for c in res.checks if c.status == "fail"]
    K = res.arx.to_tf().dc_gain
    assert abs(K - 30.0) / 30.0 < 0.03
    assert abs(res.arx.tau - 0.04) / 0.04 < 0.05


def test_generic_csv_roundtrip(tmp_path):
    n = 500
    data = SystemData(t=np.arange(n) * 0.001,
                      u=np.linspace(0, 1, n),
                      y=np.linspace(0, 2, n),
                      fs=1000.0, input_unit="V", output_unit="degC",
                      meta={"note": "roundtrip"})
    path = tmp_path / "generic.csv"
    save_system_csv(path, data)
    header = path.read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("t_s,u,y") for line in header)
    again = load_system_csv(path)
    assert again.input_unit == "V"
    assert again.output_unit == "degC"
    assert again.meta["note"] == "roundtrip"
    np.testing.assert_allclose(again.u, data.u, rtol=1e-5)
    np.testing.assert_allclose(again.y, data.y, rtol=1e-5)


def test_thresholds_configurable(second_order):
    data, _ = second_order
    res_default = run_identification(data, max_order=3)
    assert all(c.status != "fail" for c in res_default.checks)

    strict = Thresholds(vaf_train=101.0)          # impossible: always fails
    res_strict = run_identification(data, max_order=3, thresholds=strict)
    failed = {c.name for c in res_strict.checks if c.status == "fail"}
    assert any(name.startswith("1 ") for name in failed)   # VAF check trips


def test_transfer_function_outputs(second_order):
    data, truth = second_order
    res = run_identification(data, max_order=3)
    tf = res.arx.to_tf()
    assert tf.char_eq_z().endswith("= 0")
    assert tf.char_eq_s().endswith("= 0")
    # DC gain consistency of the frequency response
    h0 = tf.freq_response([0.0])[0]
    assert abs(h0 - tf.dc_gain) / abs(tf.dc_gain) < 1e-6
    assert tf.is_stable
    d = tf.to_dict()
    assert d["order"]["n"] == res.arx.n_order
    assert "G(s)_bilinear" in d and d["delay_samples"] == DELAY


def test_model_json_sections(tmp_path, second_order):
    from simforge.id.report import write_report
    data, _ = second_order
    res = run_identification(data, max_order=3)
    out = tmp_path / "rep"
    write_report(res, out)
    model = json.loads((out / "model.json").read_text(encoding="utf-8"))
    assert model["structure"].startswith("ARX(2,2)")
    assert model["order_selection"]["chosen_n"] == 2
    assert model["units"] == {"input": "V", "output": "rpm",
                              "sample_rate_hz": 500.0}
    assert len(model["characteristic_equation"]["s"]) > 5


def test_cli_end_to_end(tmp_path):
    csv = tmp_path / "cli.csv"
    generate_second_order_dataset(csv, fn_hz=FN, zeta=ZETA, gain=GAIN,
                                  ts=TS, delay=DELAY, seed=7)
    out = tmp_path / "cli_out"
    proc = subprocess.run(
        [sys.executable, str(REPO / "examples" / "identify.py"),
         "--input", str(csv), "--max-order", "3", "--outdir", str(out),
         "--input-unit", "V", "--output-unit", "rpm"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "ALL CHECKS PASSED" in proc.stdout
    assert (out / "report.html").exists()
    assert (out / "model.json").exists()
    assert len(list(out.glob("*.png"))) == 11
