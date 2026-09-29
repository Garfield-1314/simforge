"""Generate a synthetic brushed-DC motor CSV with known ground truth.

用法:
    python examples/simulate_motor.py                       # 默认 -> data/motor_sim.csv
    python examples/simulate_motor.py --out data/demo.csv --seed 7
    python examples/simulate_motor.py --deadzone 8          # 演示死区非线性检测
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

EXAMPLE_DIR = Path(__file__).resolve().parent
ROOT = EXAMPLE_DIR.parents[1]          # repo root: examples/motor -> repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simforge.id.simulator import (TRUE_K, TRUE_TAU, TRUE_TD,  # noqa: E402
                                   generate_dataset)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="generate synthetic motor data CSV")
    ap.add_argument("--out", default=str(EXAMPLE_DIR / "data" / "motor_sim.csv"))
    ap.add_argument("--fs", type=float, default=500.0)
    ap.add_argument("--cpr", type=int, default=4096, help="1024 lines x4")
    ap.add_argument("--K", type=float, default=TRUE_K, help="true gain rpm/%")
    ap.add_argument("--tau", type=float, default=TRUE_TAU, help="true tau [s]")
    ap.add_argument("--td", type=float, default=TRUE_TD, help="true delay [s]")
    ap.add_argument("--noise", type=float, default=15.0, help="process noise rpm rms")
    ap.add_argument("--deadzone", type=float, default=0.0, help="dead zone [% PWM]")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = generate_dataset(args.out, fs=args.fs, cpr=args.cpr, K=args.K,
                            tau=args.tau, td=args.td, noise_rpm=args.noise,
                            deadzone_pct=args.deadzone, seed=args.seed)

    print(f"wrote {args.out}")
    print(f"  samples : {data.n} @ {data.fs:g} Hz ({data.duration:.1f} s)")
    print(f"  encoder : {data.counts_per_rev} counts/rev")
    print(f"  ground truth: K={args.K:g} rpm/%  tau={args.tau:g} s  td={args.td:g} s")
    peak = max(abs(data.speed_rpm))
    print(f"  peak speed   : {peak:.0f} rpm "
          f"(quantisation noise ~{0.5 * data.fs * 60 / data.counts_per_rev:.1f} rpm)")


if __name__ == "__main__":
    main()
