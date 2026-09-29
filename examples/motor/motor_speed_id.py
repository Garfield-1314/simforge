"""End-to-end motor speed-loop identification.

用法:
    python examples/motor_speed_id.py --input data/motor_sim.csv
    python examples/motor_speed_id.py --input run1.csv --holdout 50 --outdir output/run1

流程: 加载 CSV -> 分段 -> ARX + PEM 双方法辨识 -> 五层验证 -> 输出报告
(8 张图 PNG + 交互式 HTML + report.md + model.json)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

EXAMPLE_DIR = Path(__file__).resolve().parent
ROOT = EXAMPLE_DIR.parents[1]          # repo root: examples/motor -> repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simforge.id import check_data, load_motor_csv, run_identification  # noqa: E402
from simforge.id.report import write_report  # noqa: E402


def print_summary(res) -> None:
    arx, pem, m = res.arx, res.pem, res.metrics
    print("=" * 72)
    print("模型  G(s) = K · exp(-td·s) / (τ·s + 1)")
    print("-" * 72)
    print(f"{'参数':<14}{'ARX (主)':>16}{'PEM (计数)':>16}{'ARX 95% CI':>24}")
    print(f"{'K [rpm/%]':<14}{arx.K:>16.2f}{pem.K:>16.2f}"
          f"{'[' + format(arx.K_ci[0], '.2f') + ', ' + format(arx.K_ci[1], '.2f') + ']':>24}")
    print(f"{'τ [ms]':<14}{arx.tau * 1e3:>16.2f}{pem.tau * 1e3:>16.2f}"
          f"{'[' + format(arx.tau_ci[0] * 1e3, '.2f') + ', ' + format(arx.tau_ci[1] * 1e3, '.2f') + ']':>24}")
    print(f"{'t_d [ms]':<14}{arx.td * 1e3:>16.1f}{pem.td * 1e3:>16.1f}{'—':>24}")
    print(f"{'带宽 [Hz]':<14}{arx.bandwidth_hz:>16.2f}{pem.bandwidth_hz:>16.2f}{'—':>24}")
    print(f"{'离散 (a,b,d)':<14}{arx.a:>16.5f}{arx.b:>16.5f}{'d = ' + str(arx.delay):>24}")
    print("-" * 72)
    print(f"训练段 VAF: ARX {m['vaf_train_arx']:.1f}%  PEM {m['vaf_train_pem']:.1f}%"
          f"   RMSE: {m['rmse_train_arx']:.1f} / {m['rmse_train_pem']:.1f} rpm")
    if res.has_holdout:
        print(f"留出段 VAF: ARX {m['vaf_holdout_arx']:.1f}%  PEM {m['vaf_holdout_pem']:.1f}%"
              f"   稳态误差: {m['ss_err_holdout_arx_pct']:.2f}% / "
              f"{m['ss_err_holdout_pem_pct']:.2f}%")
    print(f"Ljung-Box p={m['ljung_box_p']:.3f}   "
          f"残差互相关 max|ρ|={m['xcorr_max']:.4f} (带 {m['xcorr_band']:.4f})")
    print("验收清单:")
    icons = {"pass": "[OK ]", "fail": "[FAIL]", "skip": "[SKIP]", "manual": "[TODO]"}
    for c in res.checks:
        detail = c.value or c.note
        print(f"  {icons[c.status]} {c.name:<38} {detail}")
    if "k_err_pct" in m:
        print("-" * 72)
        print(f"参数恢复误差 (对照仿真真值): K {m['k_err_pct']:.2f}%  "
              f"τ {m['tau_err_pct']:.2f}%  t_d {m['td_err_s'] * 1e3:.1f} ms"
              f"   (PEM: K {m['k_err_pct_pem']:.2f}%, τ {m['tau_err_pct_pem']:.2f}%)")
    failed = [c for c in res.checks if c.status == "fail"]
    print("=" * 72)
    print(f"result: {'ALL CHECKS PASSED' if not failed else str(len(failed)) + ' CHECK(S) FAILED'}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="motor speed-loop identification")
    ap.add_argument("--input", default=str(EXAMPLE_DIR / "data" / "motor_sim.csv"))
    ap.add_argument("--outdir", default=None,
                    help="default: examples/motor/output/<input stem>")
    ap.add_argument("--holdout", type=float, default=50.0,
                    help="duty %% held out for validation (0 = disable)")
    ap.add_argument("--max-delay", type=int, default=None,
                    help="max pure delay to consider, in samples")
    ap.add_argument("--lags", type=int, default=10, help="Ljung-Box lags")
    args = ap.parse_args()

    outdir = args.outdir or str(EXAMPLE_DIR / "output" / Path(args.input).stem)

    data = load_motor_csv(args.input)
    warnings = check_data(data)
    print(f"loaded {data.source}: {data.n} samples @ {data.fs:g} Hz, "
          f"{data.duration:.1f} s, cpr={data.counts_per_rev}")
    for w in warnings:
        print(f"  [warn] {w}")

    holdout = args.holdout if args.holdout != 0 else None
    res = run_identification(data, holdout_duty=holdout,
                             max_delay=args.max_delay, lags=args.lags)
    print_summary(res)

    paths = write_report(res, outdir)
    print(f"报告: {paths['html']}")
    print(f"      {paths['md']}")
    print(f"      {Path(paths['outdir']) / 'model.json'}  (+ {len(paths['pngs'])} 张 PNG)")


if __name__ == "__main__":
    main()
