"""System I/O: generic SISO experiment data (CSV load/save/sanity checks).

CSV format
----------
* ``# key: value`` comment lines carry metadata (sample rate, units,
  encoder resolution, operating conditions ...).
* One header row, then comma separated rows.  Two layouts are supported::

      # fs: 500
      t_s,u,y            <- generic:    input column u, output column y
      0.000000,0.0,0.0

      # fs: 500
      # counts_per_rev: 4096
      t_s,pwm_pct,enc_delta   <- motor legacy: pulses are converted to rpm
      0.000000,0.0,0

Column detection is by aliases (see ``U_ALIASES`` / ``Y_ALIASES``), so
most reasonable header names work out of the box; ``load_motor_csv`` is
the motor preset of :func:`load_system_csv`.

Transforms
----------
If the output column looks like raw encoder pulses *and* the metadata
contains ``counts_per_rev``, the loader converts counts/sample to rpm and
keeps the raw column in :attr:`SystemData.raw` (round-trip safe).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TIME_ALIASES = ("t_s", "time", "t", "timestamp")

#: input (control variable) column aliases
U_ALIASES = ("u", "input", "control", "pwm_pct", "pwm", "duty", "duty_pct",
             "duty_cycle", "voltage", "v_in", "setpoint_cmd")
#: output (measured variable) column aliases
Y_ALIASES = ("y", "output", "enc_delta", "enc_counts", "counts", "pulses",
             "encoder", "enc", "speed", "rpm", "temperature", "value", "measured")

#: aliases that mean "raw encoder pulses per period" (trigger enc->rpm)
ENC_RAW_ALIASES = ("enc_delta", "enc_counts", "counts", "pulses", "encoder", "enc")

# legacy names used by the original motor code
PWM_ALIASES = U_ALIASES
ENC_ALIASES = Y_ALIASES


class DataFormatError(ValueError):
    """Raised when a data file cannot be parsed or is invalid."""


@dataclass
class SystemData:
    """One SISO experiment run: input ``u`` and output ``y`` over time.

    ``y`` is always in physical units (e.g. rpm, degC).  When the raw
    column was encoder pulses, the conversion result lives in ``y`` and
    the untouched counts stay in ``raw``.
    """

    t: np.ndarray
    u: np.ndarray
    y: np.ndarray
    fs: float
    input_unit: str = ""
    output_unit: str = ""
    meta: dict = field(default_factory=dict)
    source: str = ""
    raw: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.u = np.asarray(self.u, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.raw is not None:
            self.raw = np.asarray(self.raw, dtype=float)
        if not (len(self.t) == len(self.u) == len(self.y)):
            raise DataFormatError("t / u / y columns must have equal length")
        if self.raw is not None and len(self.raw) != len(self.t):
            raise DataFormatError("raw column length mismatch")
        if len(self.t) < 20:
            raise DataFormatError(f"too few samples: {len(self.t)}")
        for name, arr in (("t", self.t), ("u", self.u), ("y", self.y)):
            if not np.all(np.isfinite(arr)):
                raise DataFormatError(f"non-finite values in '{name}' column")
        if np.any(np.diff(self.t) <= 0):
            raise DataFormatError("time axis must be strictly increasing")
        if not self.fs > 0:
            raise DataFormatError("sample rate fs must be positive")

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
    def counts_per_rev(self) -> int | None:
        cpr = self.meta.get("counts_per_rev") or self.meta.get("cpr")
        try:
            return int(float(cpr)) if cpr is not None else None
        except ValueError:
            return None

    # ---- legacy motor-style accessors (kept for examples/tests) -------- #
    @property
    def pwm(self) -> np.ndarray:
        """(compat) input sequence."""
        return self.u

    @property
    def speed_rpm(self) -> np.ndarray:
        """(compat) output sequence -- physical units, rpm for motor data."""
        return self.y

    @property
    def enc_delta(self) -> np.ndarray | None:
        """(compat) raw per-period encoder pulses, if present."""
        return self.raw

    @property
    def counts_cumulative(self) -> np.ndarray | None:
        """(compat) cumulative encoder counts (position), if raw present."""
        if self.raw is None:
            return None
        return np.cumsum(self.raw)

    def meta_number(self, key: str, default=None):
        """Return a numeric metadata value, or *default* when absent."""
        raw = self.meta.get(key)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            return default


#: backward-compatible alias -- the original name of the data container
MotorData = SystemData


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


def counts_to_speed_rpm(enc: np.ndarray, fs: float, counts_per_rev: float) -> np.ndarray:
    """Per-period pulse counts -> rpm (motor helper)."""
    return np.asarray(enc, dtype=float) * (60.0 * fs / counts_per_rev)


def load_system_csv(path: str | Path, preset: str | None = None,
                    u_aliases: tuple[str, ...] | None = None,
                    y_aliases: tuple[str, ...] | None = None) -> SystemData:
    """Load a SISO experiment CSV (see module docstring for format).

    ``preset="motor"`` applies motor conventions (default units, encoder
    pulse -> rpm conversion).  ``u_aliases`` / ``y_aliases`` override the
    default column-name detection.
    """
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
    i_u = _pick(header, u_aliases or U_ALIASES, "input")
    i_y = _pick(header, y_aliases or Y_ALIASES, "output")

    try:
        table = np.array([[float(x) for x in r.split(",")] for r in rows], dtype=float)
    except ValueError as exc:
        raise DataFormatError(f"{path}: failed to parse numeric rows ({exc})") from exc
    if table.shape[1] < len(header):
        raise DataFormatError(f"{path}: ragged CSV rows")

    t = table[:, i_t]
    u = table[:, i_u]
    y_col = table[:, i_y]
    y_name = header[i_y]

    # ---- sample rate ---------------------------------------------------- #
    fs = meta.get("fs") or meta.get("sample_rate") or meta.get("sample_rate_hz")
    try:
        fs_val = float(fs) if fs is not None else 0.0
    except ValueError:
        fs_val = 0.0
    dt = np.diff(t)
    fs_est = 1.0 / float(np.median(dt))
    if fs_val <= 0:
        fs_val = fs_est
    elif abs(fs_val - fs_est) / fs_est > 0.02:
        raise DataFormatError(
            f"{path}: metadata fs={fs_val} disagrees with timestamps (fs≈{fs_est:.1f})")

    # ---- output: raw encoder pulses -> rpm when applicable --------------- #
    input_unit = meta.get("input_unit", "")
    output_unit = meta.get("output_unit", "")
    raw: np.ndarray | None = None
    y = y_col
    cpr_meta = meta.get("counts_per_rev") or meta.get("cpr") or meta.get("enc_cpr")
    is_enc_col = y_name in ENC_RAW_ALIASES
    if is_enc_col and cpr_meta is not None:
        try:
            cpr_val = float(cpr_meta)
        except ValueError as exc:
            raise DataFormatError(f"{path}: bad counts_per_rev value {cpr_meta!r}") from exc
        raw = y_col
        y = counts_to_speed_rpm(y_col, fs_val, cpr_val)
        output_unit = output_unit or "rpm"
    elif is_enc_col and preset == "motor":
        raise DataFormatError(f"{path}: missing 'counts_per_rev' metadata")

    if preset == "motor":
        input_unit = input_unit or "% PWM"
        output_unit = output_unit or "rpm"

    return SystemData(t=t, u=u, y=y, fs=fs_val, input_unit=input_unit,
                      output_unit=output_unit, meta=meta, source=str(path),
                      raw=raw)


def load_motor_csv(path: str | Path) -> SystemData:
    """Motor preset of :func:`load_system_csv` (legacy name kept)."""
    return load_system_csv(path, preset="motor")


def save_system_csv(path: str | Path, data: SystemData,
                    extra_meta: dict | None = None) -> Path:
    """Write a :class:`SystemData` back to CSV.

    Motor data (with raw pulses) keeps the legacy three-column layout so
    ``load_motor_csv`` round-trips; generic data uses ``t_s,u,y``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"fs": f"{data.fs:g}"}
    if data.counts_per_rev is not None:
        meta["counts_per_rev"] = str(data.counts_per_rev)
    meta.update({k: str(v) for k, v in data.meta.items()})
    if data.input_unit:
        meta.setdefault("input_unit", data.input_unit)
    if data.output_unit:
        meta.setdefault("output_unit", data.output_unit)
    if extra_meta:
        meta.update({k: str(v) for k, v in extra_meta.items()})

    is_motor_layout = data.raw is not None
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for key, value in meta.items():
            fh.write(f"# {key}: {value}\n")
        if is_motor_layout:
            fh.write("t_s,pwm_pct,enc_delta\n")
            rows = ((f"{ti:.6f},{ui:.4f},{int(round(ei))}")
                    for ti, ui, ei in zip(data.t, data.u, data.raw))
        else:
            fh.write("t_s,u,y\n")
            rows = ((f"{ti:.6f},{ui:.6g},{yi:.6g}")
                    for ti, ui, yi in zip(data.t, data.u, data.y))
        for row in rows:
            fh.write(row + "\n")
    return path


def save_motor_csv(path: str | Path, data: SystemData,
                   extra_meta: dict | None = None) -> Path:
    """Legacy name for :func:`save_system_csv`."""
    return save_system_csv(path, data, extra_meta)


def check_data(data: SystemData) -> list[str]:
    """Return human readable warnings (empty == data looks fine)."""
    warnings: list[str] = []
    dt = np.diff(data.t)
    med = float(np.median(dt))
    jitter = float(np.max(np.abs(dt - med)))
    if med > 0 and jitter > 0.02 * med:
        warnings.append(
            f"sampling jitter {jitter * 1e3:.2f} ms exceeds 2% of period "
            f"({med * 1e3:.2f} ms)")
    if np.all(data.u == data.u[0]):
        warnings.append("input u is constant for the whole recording "
                        "-- no excitation, nothing to identify")
    if np.ptp(data.y) <= 0:
        warnings.append("output y is constant -- system did not respond")
    return warnings
