"""Generic SISO system identification CLI.

用法:
    python examples/identify.py --input data.csv --max-order 3
    python examples/identify.py --input data.csv --holdout 10 --outdir output/run1
    python examples/identify.py --input data.csv --input-unit V --output-unit degC

CSV: `# fs: 500` metadata + header `t_s,u,y` (column aliases accepted).
Output: 11 PNGs + interactive report.html + report.md + model.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simforge.id import check_data, load_system_csv, run_identification  # noqa: E402
from simforge.id.report import write_report  # noqa: E402


def print_summary(res) -> None:
    arx, pem, m = res.arx, res.pem, res.metrics
    tf = arx.to_tf()
    data = res.data
    print("=" * 72)
    print(f"结构: ARX({arx.n_order},{arx.m_order}) + {arx.delay} 拍延迟"
          f"  |  估计器: {arx.method.upper()}  |  阶次准则: {str(m.get('criterion','')).upper()}")
    print("-" * 72)
    print(tf.g_s_str())
    print(tf.g_z_str())
    print(f"特征方程(z): {tf.char_eq_z()}")
    print(f"特征方程(s): {tf.char_eq_s()}")
    print(f"DC 增益 = {tf.dc_gain:.6g} "
          f"{data.output_unit + '/' + data.input_unit if data.input_unit and data.output_unit else ''}")
    print(f"极点(z) = {np.round(tf.poles_z(), 5).tolist()}  稳定 = {tf.is_stable}")
    print(f"极点(s) = {[complex(np.round(p, 3)) for p in tf.poles_s()]}")
    if arx.n_order == 1 and arx.tau:
        print(f"τ = {arx.tau * 1e3:.2f} ms (ARX) / {pem.tau * 1e3:.2f} ms (PEM)"
              f"   带宽 = {arx.bandwidth_hz:.3f} Hz")
    elif tf.tau_eff():
        print(f"主导时间常数 = {tf.tau_eff() * 1e3:.2f} ms"
              f"   主导带宽 = {1 / (2 * np.pi * tf.tau_eff()):.3f} Hz")
    print("-" * 72)
    print(f"训练段 VAF: ARX {m['vaf_train_arx']:.1f}% / PEM {m['vaf_train_pem']:.1f}%")
    if res.has_holdout:
        print(f"留出段 VAF: ARX {m['vaf_holdout_arx']:.1f}% / PEM {m['vaf_holdout_pem']:.1f}%"
              f"   稳态误差: {m['ss_err_holdout_arx_pct']:.2f}% / "
              f"{m['ss_err_holdout_pem_pct']:.2f}%")
    print(f"Ljung-Box p={m['ljung_box_p']:.3f}   "
          f"残差互相关 max|ρ|={m['xcorr_max']:.4f} (带 {m['xcorr_band']:.4f})")
    print(f"阶次: n={m.get('order_n')}  gap={m.get('order_gap')}")
    print("验收清单:")
    icons = {"pass": "[OK ]", "fail": "[FAIL]", "skip": "[SKIP]", "manual": "[TODO]"}
    for c in res.checks:
        detail = c.value or c.note
        print(f"  {icons[c.status]} {c.name:<44} {detail}")
    failed = [c for c in res.checks if c.status == "fail"]
    print("=" * 72)
    print("result: " + ("ALL CHECKS PASSED" if not failed
                        else f"{len(failed)} CHECK(S) FAILED"))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="generic SISO system identification")
    ap.add_argument("--input", required=True, help="CSV file (t_s,u,y)")
    ap.add_argument("--outdir", default=None,
                    help="default: output/<input stem>")
    ap.add_argument("--max-order", type=int, default=3,
                    help="max denominator order for auto selection (default 3)")
    ap.add_argument("--criterion", choices=("bic", "aic"), default="bic")
    ap.add_argument("--holdout", type=float, default=None,
                    help="input level held out for validation (e.g. 10); default none")
    ap.add_argument("--max-delay", type=int, default=None)
    ap.add_argument("--lags", type=int, default=10, help="Ljung-Box lags")
    ap.add_argument("--input-unit", default=None, help="override input unit label")
    ap.add_argument("--output-unit", default=None, help="override output unit label")
    args = ap.parse_args()

    outdir = args.outdir or str(ROOT / "output" / Path(args.input).stem)

    data = load_system_csv(args.input)
    if args.input_unit:
        data.input_unit = args.input_unit
    if args.output_unit:
        data.output_unit = args.output_unit
    warnings = check_data(data)
    print(f"loaded {data.source}: {data.n} samples @ {data.fs:g} Hz, "
          f"{data.duration:.1f} s, in={data.input_unit or 'u'}, "
          f"out={data.output_unit or 'y'}")
    for w in warnings:
        print(f"  [warn] {w}")

    res = run_identification(data, holdout_duty=args.holdout,
                             max_delay=args.max_delay, lags=args.lags,
                             max_order=args.max_order,
                             criterion=args.criterion)
    print_summary(res)

    paths = write_report(res, outdir,
                         title=f"系统辨识报告 — {Path(args.input).name}")
    print(f"报告: {paths['html']}")
    print(f"      {paths['md']}")
    print(f"      {Path(paths['outdir']) / 'model.json'}  (+ {len(paths['pngs'])} 张 PNG)")


if __name__ == "__main__":
    main()
