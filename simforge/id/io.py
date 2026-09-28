"""Motor experiment data: CSV loading, saving and sanity checks.

CSV format
----------
* ``# key: value`` comment lines carry metadata (sample rate, encoder
  resolution, supply voltage, operating conditions ...).
* One header row, then comma separated rows::

      t_s,pwm_pct,enc_delta
      0.000000,0.0,0
      0.002000,0.0,0

``enc_delta`` is the number of encoder pulses accumulated during that
sampling period (i.e. a velocity measure in counts/sample).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TIME_ALIASES = ("t_s", "time", "t", "timestamp")
PWM_ALIASES = ("pwm_pct", "pwm", "duty", "duty_pct", "duty_cycle")
ENC_ALIASES = ("enc_delta", "enc_counts", "counts", "pulses", "encoder", "enc")


class DataFormatError(ValueError):
    """Raised when a motor data file cannot be parsed or is invalid."""


@dataclass
class MotorData:
    """One motor experiment run."""

    t: np.ndarray
    pwm: np.ndarray
    enc_delta: np.ndarray
    fs: float
    counts_per_rev: int
    meta: dict = field(default_factory=dict)
    source: str = ""

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.pwm = np.asarray(self.pwm, dtype=float)
        self.enc_delta = np.asarray(self.enc_delta, dtype=float)
        if not (len(self.t) == len(self.pwm) == len(self.enc_delta)):
            raise DataFormatError("t / pwm / enc columns must have equal length")
        if len(self.t) < 20:
            raise DataFormatError(f"too few samples: {len(self.t)}")
        for name, arr in (("t", self.t), ("pwm", self.pwm), ("enc", self.enc_delta)):
            if not np.all(np.isfinite(arr)):
                raise DataFormatError(f"non-finite values in '{name}' column")
        if np.any(np.diff(self.t) <= 0):
            raise DataFormatError("time axis must be strictly increasing")
        if not self.fs > 0:
            raise DataFormatError("sample rate fs must be positive")
        if not self.counts_per_rev > 0:
            raise DataFormatError("counts_per_rev must be positive")

    # ------------------------------------------------------------------ #
    @property
    def ts(self) -> float:
        """Sampling period [s]."""
        return 1.0 / self.fs

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0] + self.ts)

    @property
    def speed_rpm(self) -> np.ndarray:
        """Speed in rpm derived from per-period pulse counts."""
        return self.enc_delta * (60.0 * self.fs / self.counts_per_rev)

    @property
    def counts_cumulative(self) -> np.ndarray:
        """Cumulative encoder counts (position) since start of recording."""
        return np.cumsum(self.enc_delta)

    def meta_number(self, key: str, default=None):
        """Return a numeric metadata value, or *default* when absent."""
        raw = self.meta.get(key)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            return default


# --------------------------------------------------------------------------- #
def _parse_meta_line(body: str) -> tuple[str, str] | None:
    if ":" in body:
        key, value = body.split(":", 1)
    elif "=" in body:
        key, value = body.split("=", 1)
    else:
        return None
    key = key.strip().lower()
    value = value.strip()
    if key:
        return key, value
    return None


def _pick(header: list[str], aliases: tuple[str, ...], what: str) -> int:
    for name in aliases:
        if name in header:
            return header.index(name)
    raise DataFormatError(f"cannot find {what} column (aliases: {aliases}; got {header})")


def load_motor_csv(path: str | Path) -> MotorData:
    """Load a motor experiment CSV file (see module docstring for format)."""
    path = Path(path)
    meta: dict[str, str] = {}
    header: list[str] | None = None
    rows: list[str] = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                parsed = _parse_meta_line(line.lstrip("#").strip())
                if parsed is not None:
                    meta[parsed[0]] = parsed[1]
                continue
            if header is None:
                header = [c.strip().lower() for c in line.split(",")]
            else:
                rows.append(line)
    if header is None or not rows:
        raise DataFormatError(f"{path}: no CSV data found")

    i_t = _pick(header, TIME_ALIASES, "time")
    i_pwm = _pick(header, PWM_ALIASES, "PWM")
    i_enc = _pick(header, ENC_ALIASES, "encoder")

    try:
        table = np.array([[float(x) for x in r.split(",")] for r in rows], dtype=float)
    except ValueError as exc:
        raise DataFormatError(f"{path}: failed to parse numeric rows ({exc})") from exc
    if table.shape[1] < len(header):
        raise DataFormatError(f"{path}: ragged CSV rows")

    t = table[:, i_t]
    pwm = table[:, i_pwm]
    enc = table[:, i_enc]

    # ---- sample rate -------------------------------------------------- #
    fs = meta.get("fs") or meta.get("sample_rate") or meta.get("sample_rate_hz")
    if fs is not None:
        try:
            fs_val = float(fs)
        except ValueError:
            fs_val = 0.0
    else:
        fs_val = 0.0
    dt = np.diff(t)
    fs_est = 1.0 / float(np.median(dt))
    if fs_val <= 0:
        fs_val = fs_est
    elif abs(fs_val - fs_est) / fs_est > 0.02:
        raise DataFormatError(
            f"{path}: metadata fs={fs_val} disagrees with timestamps (fs≈{fs_est:.1f})"
        )

    # ---- encoder resolution ------------------------------------------- #
    cpr = meta.get("counts_per_rev") or meta.get("cpr") or meta.get("enc_cpr")
    if cpr is None:
        raise DataFormatError(f"{path}: missing 'counts_per_rev' metadata")
    try:
        cpr_val = int(float(cpr))
    except ValueError as exc:
        raise DataFormatError(f"{path}: bad counts_per_rev value {cpr!r}") from exc

    return MotorData(t=t, pwm=pwm, enc_delta=enc, fs=fs_val,
                     counts_per_rev=cpr_val, meta=meta, source=str(path))


def save_motor_csv(path: str | Path, data: MotorData,
                   extra_meta: dict | None = None) -> Path:
    """Write a :class:`MotorData` back to CSV (metadata first)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"fs": f"{data.fs:g}", "counts_per_rev": str(data.counts_per_rev)}
    meta.update({k: str(v) for k, v in data.meta.items()})
    if extra_meta:
        meta.update({k: str(v) for k, v in extra_meta.items()})
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for key, value in meta.items():
            fh.write(f"# {key}: {value}\n")
        fh.write("t_s,pwm_pct,enc_delta\n")
        for ti, ui, ei in zip(data.t, data.pwm, data.enc_delta):
            fh.write(f"{ti:.6f},{ui:.4f},{int(round(ei))}\n")
    return path


def check_data(data: MotorData) -> list[str]:
    """Return a list of human readable warnings (empty == data looks fine)."""
    warnings: list[str] = []
    dt = np.diff(data.t)
    med = float(np.median(dt))
    jitter = float(np.max(np.abs(dt - med)))
    if med > 0 and jitter > 0.02 * med:
        warnings.append(
            f"sampling jitter {jitter * 1e3:.2f} ms exceeds 2% of period "
            f"({med * 1e3:.2f} ms)")
    if np.max(np.abs(data.pwm)) > 100.0:
        warnings.append(f"PWM outside [-100, 100]: [{np.min(data.pwm)}, {np.max(data.pwm)}]")
    if np.all(data.enc_delta == 0):
        warnings.append("encoder reports zero pulses for the whole recording")
    speed = data.speed_rpm
    if np.max(np.abs(speed)) < 50:
        warnings.append(f"peak speed only {np.max(np.abs(speed)):.1f} rpm -- "
                        "too slow for a reliable identification")
    return warnings
