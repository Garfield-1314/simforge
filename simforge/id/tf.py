"""Transfer function utilities: G(z), G(s), characteristic equation, poles/zeros.

Model convention (matches :mod:`simforge.id.arx`) -- an ARX(n,m) model
with pure delay ``d`` (samples)::

    y[k+1] = sum_i a_i * y[k-i]  +  sum_j b_j * u[k-d-j]

which in the z-domain is::

              z^-(d+1) * (b0 + b1 z^-1 + ... + b_{m-1} z^-(m-1))
    G(z) = ---------------------------------------------------------
                1 - a0 z^-1 - a1 z^-2 - ... - a_{n-1} z^-n

* ``G(z)`` and the z-domain characteristic equation are **exact**.
* ``G(s)`` is obtained by the inverse bilinear (Tustin) transform
  ``z = (2 + sTs)/(2 - sTs)`` -- an approximation noted as such;
  continuous *poles* are mapped exactly via ``s = ln(z)/Ts`` (exact for
  the sampled poles of a ZOH-discretised stable/unstable mode).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import Polynomial


# --------------------------------------------------------------------------- #
def _fmt_num(x: float, digits: int = 4) -> str:
    if x == 0:
        return "0"
    return f"{x:.{digits}g}"


def poly_str(coeffs_desc: np.ndarray, var: str = "z") -> str:
    """Format a polynomial given as descending powers: '2 z^2 - 1.5 z + 0.3'."""
    c = np.asarray(coeffs_desc, dtype=float)
    if c.size == 0:
        return "0"
    # strip leading zeros but keep at least one coefficient
    idx = np.flatnonzero(np.abs(c) > 1e-12)
    if idx.size == 0:
        return "0"
    c = c[idx[0]:]
    deg = len(c) - 1
    terms = []
    for i, coef in enumerate(c):
        power = deg - i
        if abs(coef) < 1e-12:
            continue
        sign = "-" if coef < 0 else ("+" if terms else "")
        mag = abs(coef)
        if power == 0:
            body = _fmt_num(mag)
        elif power == 1:
            body = ("" if abs(mag - 1) < 1e-12 else _fmt_num(mag)) + f" {var}"
        else:
            body = ("" if abs(mag - 1) < 1e-12 else _fmt_num(mag)) + f" {var}^{power}"
        terms.append((sign + body) if sign else body)
    return " ".join(terms) if terms else "0"


def _stable_complex_list(roots: np.ndarray) -> list[list[float]]:
    return [[float(np.real(r)), float(np.imag(r))] for r in roots]


@dataclass
class TransferFunction:
    """Discrete transfer function G(z) of an identified ARX model."""

    a: np.ndarray                  # a_i (n coefficients), y[k+1] recursion
    b: np.ndarray                  # b_j (m coefficients)
    delay: int                     # d (samples);  y[k+1] uses u[k-d]
    ts: float
    input_unit: str = ""
    output_unit: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.a = np.atleast_1d(np.asarray(self.a, dtype=float))
        self.b = np.atleast_1d(np.asarray(self.b, dtype=float))

    # ---- basic orders --------------------------------------------------- #
    @property
    def n_order(self) -> int:
        """Denominator order n."""
        return len(self.a)

    @property
    def m_order(self) -> int:
        """Numerator order m (number of input taps)."""
        return len(self.b)

    # ---- z-domain polynomials (scipy.signal.dlti convention) ------------ #
    def num_den_z(self) -> tuple[np.ndarray, np.ndarray]:
        """(numerator, denominator) as descending powers of z.

        The pure delay is folded into the polynomials (numerator shift +
        poles at the origin in the denominator), so evaluating the ratio
        at z = e^{jwTs} gives the exact frequency response including delay.
        """
        n, m, d = self.n_order, self.m_order, int(self.delay)
        S = max(n, d + m)
        num = np.zeros(S + 1)
        for j, bj in enumerate(self.b):
            power = S - d - 1 - j          # b_j sits at z^(S-d-1-j)
            if power >= 0:
                num[S - power] += bj
        den = np.zeros(S + 1)
        den[0] = 1.0                       # z^S
        for i, ai in enumerate(self.a):
            # a_i multiplies z^(S-i-1) -> array index i+1
            den[i + 1] -= ai
        return num, den

    # ---- properties ------------------------------------------------------ #
    @property
    def dc_gain(self) -> float:
        """G(z=1) = sum(b) / (1 - sum(a))  [output units per input unit]."""
        s = 1.0 - float(np.sum(self.a))
        if abs(s) < 1e-12:
            return float("inf")
        return float(np.sum(self.b) / s)

    @property
    def K(self) -> float:
        """(compat) DC gain."""
        return self.dc_gain

    def poles_z(self) -> np.ndarray:
        _, den = self.num_den_z()
        return np.roots(den)

    def zeros_z(self) -> np.ndarray:
        num, _ = self.num_den_z()
        num = num[np.flatnonzero(np.abs(num) > 1e-12)]
        if num.size <= 1:
            return np.empty(0)
        return np.roots(num)

    def poles_s(self) -> np.ndarray:
        """Continuous poles via s = ln(z)/Ts (poles at 0 -> -inf delay)."""
        pz = self.poles_z()
        out = []
        for z in pz:
            if abs(z) < 1e-12:
                out.append(complex(-np.inf, 0.0))
            else:
                out.append(np.log(complex(z)) / self.ts)
        return np.array(out, dtype=complex)

    @property
    def is_stable(self) -> bool:
        return bool(np.all(np.abs(self.poles_z()) < 1.0 - 1e-12))

    def tau_eff(self) -> float | None:
        """Dominant continuous time constant [s]; order-1 -> exact tau."""
        ps = self.poles_s()
        finite = ps[np.isfinite(ps.real)]
        if finite.size == 0:
            return None
        # dominant = slowest stable pole (closest to the imaginary axis)
        dom = finite[np.argmin(np.abs(finite.real))]
        if dom.real >= 0:
            return None                      # unstable / marginally stable
        return float(-1.0 / dom.real)

    def td(self) -> float:
        """Pure delay parameter [s] (parameter convention: d * Ts)."""
        return int(self.delay) * self.ts

    # ---- frequency response ---------------------------------------------- #
    def freq_response(self, freqs_hz: np.ndarray) -> np.ndarray:
        """G(e^{j 2 pi f Ts}) -- exact, including the pure delay."""
        num, den = self.num_den_z()
        z = np.exp(1j * 2.0 * np.pi * np.asarray(freqs_hz, dtype=float) * self.ts)
        zp = np.arange(len(den) - 1, -1, -1)          # descending powers
        num_v = np.polyval(num, z)
        den_v = np.polyval(den, z)
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.where(np.abs(den_v) > 1e-300, num_v / np.where(den_v == 0, 1, den_v), np.nan)
        return h

    # ---- continuous approximation ----------------------------------------- #
    def base_num_den_z(self) -> tuple[np.ndarray, np.ndarray]:
        """G_base(z) with the explicit pure delay z^-d factored out.

        ``G(z) = z^-d * G_base(z)`` where G_base always has relative
        degree 1 (no poles at the origin), so its bilinear image carries
        no spurious delay poles.
        """
        n = self.n_order
        num = np.zeros(n)                      # degree n-1
        for j, bj in enumerate(self.b):
            if n - 1 - j >= 0:
                num[j] += bj                   # b_j at power n-1-j -> index j
        den = np.zeros(n + 1)
        den[0] = 1.0
        for i, ai in enumerate(self.a):
            den[i + 1] -= ai
        return num, den

    def num_den_s(self) -> tuple[np.ndarray, np.ndarray]:
        """(num, den) of the *rational part* in s via inverse bilinear.

        z = (2 + sTs)/(2 - sTs) applied to G_base(z); the pure delay
        z^-d is kept separately (see :meth:`delay_factor_str`).
        """
        num_z, den_z = self.base_num_den_z()
        S = len(den_z) - 1
        up = Polynomial([2.0, self.ts])        # 2 + s Ts
        dn = Polynomial([2.0, -self.ts])       # 2 - s Ts

        def to_s(arr: np.ndarray) -> Polynomial:
            # arr[i] is the coefficient of z^(S - i); multiply through by
            # (2 - sTs)^S so everything becomes a polynomial in s
            out = Polynomial([0.0])
            for i, ci in enumerate(arr):
                if ci != 0.0:
                    out += ci * (up ** (S - i)) * (dn ** i)
            return out

        ns = to_s(num_z)
        ds = to_s(den_z)
        lead = ds.coef[-1]                     # highest-degree coefficient
        if abs(lead) > 1e-14:
            ns = ns / lead
            ds = ds / lead
        return ns.coef[::-1], ds.coef[::-1]     # descending powers of s

    # ---- strings ---------------------------------------------------------- #
    @property
    def gain_unit(self) -> str:
        if self.input_unit and self.output_unit:
            return f"{self.output_unit}/{self.input_unit}"
        return ""

    def g_z_str(self) -> str:
        num, den = self.num_den_z()
        s_poly = f"({poly_str(num, 'z')})"
        s_den = f"({poly_str(den, 'z')})"
        return f"G(z) = {s_poly} / {s_den}"

    def delay_factor_str(self) -> str:
        if self.delay <= 0:
            return ""
        return f" * exp(-{self.td():.4g} s)"

    def g_s_str(self) -> str:
        ns, ds = self.num_den_s()
        return (f"G(s) = ({poly_str(ns, 's')}) / ({poly_str(ds, 's')})"
                f"{self.delay_factor_str()}")

    def char_eq_z(self) -> str:
        _, den = self.num_den_z()
        # monic
        monic = np.asarray(den, dtype=float) / den[0]
        return f"{poly_str(monic, 'z')} = 0"

    def char_eq_s(self) -> str:
        _, ds = self.num_den_s()
        ds = np.asarray(ds, dtype=float)
        nz = np.flatnonzero(np.abs(ds) > 1e-12)
        if nz.size:
            ds = ds[nz[0]:]
        monic = ds / ds[0]
        return f"{poly_str(monic, 's')} = 0"

    # ---- serialisation ----------------------------------------------------- #
    def to_dict(self) -> dict:
        pz = self.poles_z()
        zz = self.zeros_z()
        ps = self.poles_s()
        out = {
            "G(z)": self.g_z_str(),
            "G(s)_bilinear": self.g_s_str(),
            "delay_samples": int(self.delay),
            "delay_s": self.td(),
            "dc_gain": self.dc_gain,
            "gain_unit": self.gain_unit,
            "order": {"n": self.n_order, "m": self.m_order},
            "numerator_z": [float(x) for x in self.num_den_z()[0]],
            "denominator_z": [float(x) for x in self.num_den_z()[1]],
            "characteristic_equation_z": self.char_eq_z(),
            "characteristic_equation_s": self.char_eq_s(),
            "poles_z": _stable_complex_list(pz),
            "zeros_z": _stable_complex_list(zz),
            "poles_s": _stable_complex_list(ps),
            "stable_discrete": self.is_stable,
            "sample_time_s": self.ts,
        }
        tau = self.tau_eff()
        if tau is not None:
            out["dominant_tau_s"] = tau
            out["bandwidth_hz"] = 1.0 / (2.0 * np.pi * tau)
        return out


def first_order_from_arx(a0: float, b0: float, ts: float) -> tuple[float, float]:
    """Exact continuous (K, tau) of a first-order ZOH model."""
    K = b0 / (1.0 - a0)
    tau = -ts / np.log(a0)
    return float(K), float(tau)
