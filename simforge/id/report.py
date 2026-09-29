"""Report generation: 11 figures (PNG + interactive HTML), model.json, report.md.

Figure list
-----------
T1  acquisition overview (input & output, hold-out highlighted)
T2  step-response fit: measured vs model (train emphasised)
T3  hold-out prediction (out-of-sample)
T4  free-run residual timeline with 95% band
T5  ARX vs PEM vs measured (zoom on a train segment)
F1  Bode magnitude: model + Welch FRF estimate with coherence
F2  Bode phase: model + Welch FRF estimate
F3  residual power spectrum (flat == white noise)
M1  model summary card: G(s), G(z), characteristic equations, poles
M2  pole-zero map (z-plane unit circle + s-plane)
M3  order-selection sweep (criterion per order + tie tolerance band)

All labels use the units declared on the data (``input_unit`` /
``output_unit``), so the report reads correctly for any system.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

from .pipeline import IdentificationResult
from .validate import residual_psd

C_MEAS = "#333333"
C_ARX = "#1f77b4"
C_PEM = "#ff7f0e"
C_HOLD = "#ffd8b0"
C_RESID = "#9467bd"
C_ORDER = "#d62728"

CHECK_ICON = {"pass": "✅", "fail": "❌", "skip": "⏭️", "manual": "🧭"}


# --------------------------------------------------------------------------- #
# figure specs (plain dicts so matplotlib and plotly can share them)
# --------------------------------------------------------------------------- #
def _panel(series=(), spans=(), hlines=(), *, xlabel="", ylabel="",
           title="", xscale="linear", yscale="linear", xlim=None,
           ylim=None, legend=True, text=None, equal_aspect=False) -> dict:
    return {"series": list(series), "spans": list(spans), "hlines": list(hlines),
            "xlabel": xlabel, "ylabel": ylabel, "title": title,
            "xscale": xscale, "yscale": yscale, "xlim": xlim, "ylim": ylim,
            "legend": legend, "text": text, "equal_aspect": equal_aspect}


def _line(x, y, label, color, ls="-", lw=1.4, alpha=1.0, kind="line", ms=4.0):
    return {"x": np.asarray(x), "y": np.asarray(y), "label": label,
            "color": color, "ls": ls, "lw": lw, "alpha": alpha, "kind": kind,
            "ms": ms}


def _units(res: IdentificationResult) -> tuple[str, str, str]:
    """(input label, output label, gain unit)."""
    d = res.data
    in_u = d.input_unit or "u"
    out_u = d.output_unit or "y"
    gain_u = f"{out_u}/{in_u}" if (d.input_unit and d.output_unit) else ""
    return f"输入 [{in_u}]", f"输出 [{out_u}]", gain_u


def build_figures(res: IdentificationResult) -> list[dict]:
    """Build all 11 figure definitions from an identification result."""
    data = res.data
    t, y = data.t, data.y
    arx, pem, m = res.arx, res.pem, res.metrics
    in_lbl, out_lbl, gain_u = _units(res)
    n1 = arx.n_order == 1

    if n1:
        p_txt = (f"K={arx.K:.4g}{(' ' + gain_u) if gain_u else ''}, "
                 f"τ={arx.tau * 1e3:.1f} ms, t_d={arx.td * 1e3:.1f} ms")
    else:
        p_txt = (f"阶次 n={arx.n_order}, m={arx.m_order}, "
                 f"DC 增益={arx.K:.4g}{(' ' + gain_u) if gain_u else ''}, "
                 f"t_d={arx.td * 1e3:.1f} ms")

    hold_span = None
    if res.has_holdout:
        hold_span = {"x0": float(t[res.hold_idx[0]]),
                     "x1": float(t[res.hold_idx[-1]]),
                     "color": C_HOLD, "label": "留出段(未参与辨识)"}
    spans1 = [hold_span] if hold_span else []

    figs: list[dict] = []

    # ---- T1 overview ----------------------------------------------------- #
    figs.append({"name": "T1_overview", "title": "T1 采集数据总览",
                 "figsize": (10, 6),
                 "panels": [
                     _panel([_line(t, data.u, in_lbl, C_ARX)],
                            spans=spans1, xlabel="t [s]", ylabel=in_lbl),
                     _panel([_line(t, y, out_lbl, C_MEAS, lw=0.9)],
                            spans=spans1, xlabel="t [s]", ylabel=out_lbl),
                 ]})

    # ---- T2 step fit ------------------------------------------------------- #
    figs.append({"name": "T2_step_fit", "title": f"T2 拟合 — {p_txt}",
                 "figsize": (10, 4.6),
                 "panels": [
                     _panel([_line(t, y, "实测", C_MEAS, lw=0.9),
                             _line(t, res.v_pem, "PEM 模型 (OE)", C_PEM),
                             _line(t, res.v_arx, "ARX 模型", C_ARX, ls="--")],
                            spans=spans1, xlabel="t [s]", ylabel=out_lbl),
                 ]})

    # ---- T3 hold-out ------------------------------------------------------- #
    if res.has_holdout:
        i0, i1 = int(res.hold_idx[0]), int(res.hold_idx[-1])
        pad = 0.3
        xlim = (t[i0] - pad, t[i1] + pad)
        sl = slice(i0, i1 + 1)
        figs.append({"name": "T3_holdout",
                     "title": ("T3 留出段预测  VAF: "
                               f"ARX {m.get('vaf_holdout_arx', float('nan')):.1f}% / "
                               f"PEM {m.get('vaf_holdout_pem', float('nan')):.1f}%"),
                     "figsize": (10, 4.6),
                     "panels": [
                         _panel([_line(t[sl], y[sl], "实测", C_MEAS, lw=0.9),
                                 _line(t[sl], res.v_pem[sl], "PEM 模型", C_PEM),
                                 _line(t[sl], res.v_arx[sl], "ARX 模型", C_ARX, ls="--")],
                                xlabel="t [s]", ylabel=out_lbl, xlim=xlim),
                     ]})
    else:
        figs.append({"name": "T3_holdout", "title": "T3 留出段预测 (无留出段)",
                     "figsize": (10, 3.2),
                     "panels": [_panel([], xlabel="t [s]", ylabel=out_lbl)]})

    # ---- T4 residuals ------------------------------------------------------- #
    sigma = float(np.std(res.e_free))
    band = 1.96 * sigma
    figs.append({"name": "T4_residual",
                 "title": f"T4 残差时序 (实测−ARX 仿真)  1σ={sigma:.4g} "
                          f"{data.output_unit or 'y'}",
                 "figsize": (10, 4.0),
                 "panels": [
                     _panel([_line(t, res.e_free, "残差", C_RESID, lw=0.7)],
                            hlines=[{"y": band, "color": "gray", "ls": "--",
                                     "label": "±1.96σ"},
                                    {"y": -band, "color": "gray", "ls": "--"},
                                    {"y": 0.0, "color": "black", "ls": ":", "lw": 0.8}],
                            xlabel="t [s]", ylabel=f"误差 [{data.output_unit or 'y'}]"),
                 ]})

    # ---- T5 methods compare (first train segment) -------------------------- #
    pulses = [s for s in res.segments if abs(s.duty) > 1e-9]
    hold0 = res.hold_idx[0] if res.has_holdout else -1
    train_pulse = next((s for s in pulses if s.idx0 != hold0), None)
    if train_pulse is None and pulses:
        train_pulse = pulses[0]
    if train_pulse is not None:
        pad = 0.4
        xlim5 = (t[train_pulse.idx0] - pad,
                 t[min(train_pulse.idx1 + int(0.5 * data.fs), data.n - 1)] + pad)
    else:
        xlim5 = None
    figs.append({"name": "T5_methods",
                 "title": ("T5 双方法对比 (训练段)  "
                           f"K 差异 {m['k_diff_pct']:.2f}%"
                           + (f" / τ 差异 {m['tau_diff_pct']:.2f}%"
                              if np.isfinite(m.get("tau_diff_pct", float('nan'))) else "")),
                 "figsize": (10, 4.6),
                 "panels": [
                     _panel([_line(t, y, "实测", C_MEAS, lw=0.9),
                             _line(t, res.v_arx, "ARX", C_ARX, ls="--"),
                             _line(t, res.v_pem, "PEM", C_PEM, ls=":")],
                            xlabel="t [s]", ylabel=out_lbl, xlim=xlim5),
                 ]})

    # ---- F1/F2 Bode (discrete model frequency response, exact) -------------- #
    f, H, coh = res.frf
    mask = f > 0
    f_plot = f[mask]
    mag_data = 20.0 * np.log10(np.maximum(np.abs(H[mask]), 1e-12))
    phase_data = np.unwrap(np.angle(H[mask])) * 180.0 / np.pi
    good = coh[mask] >= 0.7
    f_max = min(50.0, float(f_plot.max()) if f_plot.max() > 0 else 50.0)
    freqs = np.logspace(np.log10(max(f_plot.min(), 0.05)),
                        np.log10(max(f_max, 0.1)), 400)
    tf = arx.to_tf()
    h_model = tf.freq_response(freqs)
    mag_arx = 20.0 * np.log10(np.maximum(np.abs(h_model), 1e-12))
    ph_arx = np.unwrap(np.angle(h_model)) * 180.0 / np.pi
    tfp = pem.to_tf()
    h_pem = tfp.freq_response(freqs)
    mag_pem = 20.0 * np.log10(np.maximum(np.abs(h_pem), 1e-12))
    ph_pem = np.unwrap(np.angle(h_pem)) * 180.0 / np.pi
    if n1:
        mag_ref = 20 * np.log10(max(arx.K, 1e-9)) - 3.0
        bw_txt = f"-3dB {mag_ref:.1f} dB, 带宽 {arx.bandwidth_hz:.2f} Hz"
    else:
        tau = tf.tau_eff()
        mag_ref = 20 * np.log10(max(arx.K, 1e-9)) - 3.0
        bw_txt = (f"-3dB {mag_ref:.1f} dB, 主导带宽 {1/(2*np.pi*tau):.2f} Hz"
                  if tau else f"-3dB {mag_ref:.1f} dB")

    figs.append({"name": "F1_bode_mag", "title": f"F1 伯德幅频 — {bw_txt}",
                 "figsize": (10, 4.6),
                 "panels": [
                     _panel([
                         _line(freqs, mag_arx, "ARX 模型", C_ARX),
                         _line(freqs, mag_pem, "PEM 模型", C_PEM, ls="--"),
                         _line(f_plot[good], mag_data[good], "实测 FRF (相干≥0.7)",
                               "#2ca02c", kind="scatter", ms=3, ls=""),
                         _line(f_plot[~good], mag_data[~good], "实测 FRF (低相干)",
                               "#2ca02c", kind="scatter", ms=3, ls="", alpha=0.35),
                         ], hlines=[{"y": mag_ref, "color": "gray", "ls": "--",
                                     "label": "-3dB"}],
                         xlabel="频率 [Hz]", ylabel="幅值 [dB]",
                         xscale="log", xlim=(freqs[0], freqs[-1])),
                 ]})

    figs.append({"name": "F2_bode_phase",
                 "title": "F2 伯德相频 — 模型 vs 实测 (纯滞后主导高频相位)",
                 "figsize": (10, 4.2),
                 "panels": [
                     _panel([
                         _line(freqs, ph_arx, "ARX 模型", C_ARX),
                         _line(freqs, ph_pem, "PEM 模型", C_PEM, ls="--"),
                         _line(f_plot[good], phase_data[good], "实测 FRF (相干≥0.7)",
                               "#2ca02c", kind="scatter", ms=3, ls=""),
                         _line(f_plot[~good], phase_data[~good], "实测 FRF (低相干)",
                               "#2ca02c", kind="scatter", ms=3, ls="", alpha=0.35),
                         ], xlabel="频率 [Hz]", ylabel="相位 [deg]",
                         xscale="log", xlim=(freqs[0], freqs[-1])),
                 ]})

    # ---- F3 residual PSD ---------------------------------------------------- #
    f_r, pxx = residual_psd(res.e_fit, data.fs)
    pxx_db = 10.0 * np.log10(np.maximum(pxx, 1e-20))
    floor = float(np.mean(pxx_db))
    figs.append({"name": "F3_residual_psd",
                 "title": "F3 残差功率谱 — 平坦即白噪声",
                 "figsize": (10, 4.0),
                 "panels": [
                     _panel([_line(f_r, pxx_db, "一步预测残差 PSD", C_RESID,
                                   kind="scatter", ms=3, ls="-", lw=1.0)],
                            hlines=[{"y": floor, "color": "gray", "ls": "--",
                                     "label": f"均值 {floor:.1f} dB"}],
                            xlabel="频率 [Hz]", ylabel="PSD [dB]",
                            xlim=(0, min(50.0, float(f_r[-1]) if len(f_r) else 50))),
                 ]})

    # ---- M1 model summary card --------------------------------------------- #
    tf_d = tf.to_dict()
    gap = m.get("order_gap")
    gap_txt = f"{gap:.1f}" if gap is not None and np.isfinite(gap) else "n/a (单阶次)"
    summary_lines = [
        tf.g_s_str(),
        tf.g_z_str(),
        f"特征方程(z): {tf.char_eq_z()}",
        f"特征方程(s): {tf.char_eq_s()}",
        f"DC 增益 = {tf.dc_gain:.6g} {gain_u}   延迟 = {arx.delay} 采样 ({arx.td * 1e3:.1f} ms)",
        f"极点(z) = {np.round(tf.poles_z(), 5).tolist()}   稳定 = {tf.is_stable}",
        f"极点(s) = {[complex(np.round(p, 3)) for p in tf.poles_s()]}",
        (f"τ(精确,一阶) = {arx.tau * 1e3:.2f} ms   带宽 = {arx.bandwidth_hz:.3f} Hz"
         if n1 and arx.tau else ""),
        (f"主导时间常数 = {tf.tau_eff() * 1e3:.2f} ms" if tf.tau_eff() else ""),
        f"阶次 n={arx.n_order} m={arx.m_order}  准则={str(m.get('criterion','')).upper()} "
        f"取阶 n={int(m.get('order_n', arx.n_order))}  gap={gap_txt}",
        f"估计方法 = {arx.method.upper()} (LS=过程噪声高效 / IV=测量噪声无偏)",
    ]
    figs.append({"name": "M1_model_summary",
                 "title": "M1 模型摘要 (传递函数 · 特征方程 · 极点)",
                 "figsize": (10, 4.4),
                 "panels": [_panel([], xlabel="", ylabel="", legend=False,
                                   text="\n".join(s for s in summary_lines if s))]})

    # ---- M2 pole-zero map ---------------------------------------------------- #
    pz = tf.poles_z()
    zz = tf.zeros_z()
    circ = np.exp(1j * np.linspace(0, 2 * np.pi, 361))
    ps = tf.poles_s()
    finite_s = ps[np.isfinite(ps.real)]
    figs.append({"name": "M2_pole_zero",
                 "title": "M2 零极点 (上: Z 平面 · 下: S 平面)",
                 "figsize": (10, 4.6),
                 "panels": [
                     _panel([
                         _line(circ.real, circ.imag, "|z|=1", "#999999", ls="--", lw=0.8),
                         _line(np.real(pz), np.imag(pz), "极点", C_ORDER,
                               kind="scatter", ms=9, ls=""),
                         _line(np.real(zz), np.imag(zz), "零点", "#17becf",
                               kind="scatter", ms=9, ls=""),
                         ], xlabel="Re(z)", ylabel="Im(z)", title="Z 平面",
                         xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), equal_aspect=True),
                     _panel([
                         _line(np.real(finite_s), np.imag(finite_s), "极点", C_ORDER,
                               kind="scatter", ms=9, ls=""),
                         _line([0, 0], [np.min(np.imag(finite_s)) if finite_s.size else -1,
                                       np.max(np.imag(finite_s)) if finite_s.size else 1],
                               "jw 轴", "#999999", ls="--", lw=0.8),
                         ], xlabel="Re(s) [1/s]", ylabel="Im(s) [1/s]", title="S 平面"),
                 ]})

    # ---- M3 order sweep ------------------------------------------------------ #
    ob = m.get("order_bic") or {}
    if ob:
        orders = sorted(int(k) for k in ob)
        vals = [ob[str(o)] if str(o) in ob else ob[o] for o in orders]
        best_v = min(vals)
        tol = float(m.get("order_tolerance") or 0.0)
        chosen = int(m.get("order_n", orders[0]))
        xs = [str(o) for o in orders]
        figs.append({"name": "M3_order_sweep",
                     "title": (f"M3 阶次选择 — 准则 {str(m.get('criterion','')).upper()} "
                               f"取 n={chosen} (平局容差 {tol:.0f})"),
                     "figsize": (10, 4.0),
                     "panels": [
                         _panel([
                             _line(xs, vals, "各阶次最优准则值", C_ARX,
                                   kind="scatter", ms=8, ls="-"),
                             _line([str(chosen)], [ob.get(str(chosen), best_v)],
                                   "选定阶次", C_ORDER, kind="scatter", ms=12, ls=""),
                             ], hlines=[{"y": best_v + tol, "color": "gray",
                                         "ls": "--", "label": "简单性容差 (+5% RSS)"}],
                             xlabel="阶次 n", ylabel=f"{str(m.get('criterion','')).upper()} 准则值"),
                     ]})
    else:
        figs.append({"name": "M3_order_sweep",
                     "title": "M3 阶次选择 (未记录扫描)",
                     "figsize": (10, 3.0),
                     "panels": [_panel([], xlabel="阶次 n", ylabel="准则值")]})

    return figs


# --------------------------------------------------------------------------- #
# renderers
# --------------------------------------------------------------------------- #
def _render_mpl_panel(ax, panel: dict) -> None:
    for span in panel["spans"]:
        ax.axvspan(span["x0"], span["x1"], color=span["color"], alpha=0.55,
                   label=span.get("label"))
    for s in panel["series"]:
        x, y = s["x"], s["y"]
        if len(x) == 0:
            continue
        if s["kind"] == "scatter":
            ax.scatter(x, y, s=s["ms"], c=s["color"], alpha=s["alpha"],
                       label=s["label"], zorder=3)
        else:
            ax.plot(x, y, color=s["color"], ls=s["ls"], lw=s["lw"],
                    alpha=s["alpha"], label=s["label"])
    for h in panel["hlines"]:
        ax.axhline(h["y"], color=h["color"], ls=h.get("ls", "--"),
                   lw=h.get("lw", 1.0), label=h.get("label"))
    if panel.get("text"):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.02, 0.96, panel["text"], transform=ax.transAxes,
                va="top", ha="left", fontsize=10,
                family=["Consolas", "Microsoft YaHei", "DejaVu Sans"],
                bbox=dict(boxstyle="round,pad=0.5", fc="#f7f7f7", ec="#cccccc"))
    ax.set_xscale(panel["xscale"])
    ax.set_yscale(panel["yscale"])
    if panel["xlim"]:
        ax.set_xlim(*panel["xlim"])
    if panel["ylim"]:
        ax.set_ylim(*panel["ylim"])
    if panel["xlabel"]:
        ax.set_xlabel(panel["xlabel"])
    if panel["ylabel"]:
        ax.set_ylabel(panel["ylabel"])
    if panel["title"]:
        ax.set_title(panel["title"], fontsize=10)
    ax.grid(alpha=0.3)
    if panel.get("equal_aspect"):
        ax.set_aspect("equal", adjustable="box")
    if panel["legend"] and not panel.get("text"):
        ax.legend(fontsize=8, loc="best")


def _render_png(figdef: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # CJK labels: Windows ships Microsoft YaHei; fall back gracefully elsewhere
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "Noto Sans CJK SC", "PingFang SC",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    panels = figdef["panels"]
    fig, axes = plt.subplots(len(panels), 1,
                             figsize=figdef.get("figsize", (10, 4 * len(panels))),
                             squeeze=False)
    for ax, panel in zip(axes.flat, panels):
        _render_mpl_panel(ax, panel)
    fig.suptitle(figdef["title"], fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _render_plotly(figdef: dict):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    panels = figdef["panels"]
    fig = make_subplots(rows=len(panels), cols=1,
                        shared_xaxes=False,
                        subplot_titles=[p["title"] or " " for p in panels])
    for r, panel in enumerate(panels, start=1):
        for span in panel["spans"]:
            fig.add_vrect(x0=span["x0"], x1=span["x1"], fillcolor=span["color"],
                          opacity=0.45, line_width=0, row=r, col=1)
        for s in panel["series"]:
            if len(s["x"]) == 0:
                continue
            if s["kind"] == "scatter":
                fig.add_trace(go.Scatter(x=s["x"], y=s["y"], mode="markers",
                                         marker={"size": s["ms"], "color": s["color"],
                                                 "opacity": s["alpha"]},
                                         name=s["label"], showlegend=(r == 1)),
                              row=r, col=1)
            else:
                fig.add_trace(go.Scatter(x=s["x"], y=s["y"], mode="lines",
                                         line={"color": s["color"], "dash":
                                               "dash" if s["ls"] == "--" else
                                               ("dot" if s["ls"] == ":" else "solid"),
                                               "width": s["lw"]},
                                         opacity=s["alpha"], name=s["label"],
                                         showlegend=(r == 1)),
                              row=r, col=1)
        for h in panel["hlines"]:
            fig.add_hline(y=h["y"], line_dash="dash", line_color=h.get("color", "gray"),
                          annotation_text=h.get("label"), row=r, col=1)
        if panel.get("text"):
            fig.add_annotation(text="<br>".join(
                ln.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                for ln in panel["text"].split("\n")),
                xref="x domain", yref="y domain", x=0.02, y=0.98,
                showarrow=False, align="left", valign="top",
                font=dict(family="Consolas, monospace", size=13),
                row=r, col=1)
        fig.update_xaxes(title_text=panel["xlabel"], type=panel["xscale"],
                         range=[np.log10(v) for v in panel["xlim"]]
                         if panel["xscale"] == "log" and panel["xlim"] else panel["xlim"],
                         row=r, col=1)
        fig.update_yaxes(title_text=panel["ylabel"], type=panel["yscale"],
                         range=panel["ylim"], row=r, col=1)
        if panel.get("equal_aspect"):
            fig.update_yaxes(scaleratio=1, row=r, col=1)
    fig.update_layout(title=figdef["title"], height=340 * len(panels) + 120,
                      template="plotly_white", legend={"orientation": "h"})
    return fig


# --------------------------------------------------------------------------- #
# report assembly
# --------------------------------------------------------------------------- #
def _checks_markdown(checks) -> str:
    lines = ["| # | 检查项 | 状态 | 实测值 | 阈值 | 备注 |",
             "|---|--------|------|--------|------|------|"]
    for i, c in enumerate(checks, 1):
        lines.append(f"| {i} | {c.name} | {CHECK_ICON[c.status]} {c.status} "
                     f"| {c.value} | {c.threshold} | {c.note} |")
    return "\n".join(lines)


def _checks_html(checks) -> str:
    rows = []
    for i, c in enumerate(checks, 1):
        color = {"pass": "#e6f4ea", "fail": "#fce8e6",
                 "skip": "#f1f3f4", "manual": "#fff8e1"}[c.status]
        rows.append(f"<tr style='background:{color}'><td>{i}</td><td>{c.name}</td>"
                    f"<td>{CHECK_ICON[c.status]} {c.status}</td><td>{c.value}</td>"
                    f"<td>{c.threshold}</td><td>{c.note}</td></tr>")
    return ("<table style='border-collapse:collapse;font-size:14px'>"
            "<tr><th>#</th><th>检查项</th><th>状态</th><th>实测值</th>"
            "<th>阈值</th><th>备注</th></tr>" + "".join(rows) + "</table>")


def _model_json(res: IdentificationResult) -> dict:
    m = res.metrics
    data = res.data
    tf = res.arx.to_tf()
    lower = {k.lower(): v for k, v in data.meta.items()}
    conds = {"supply_V": lower.get("supply_v"), "load": lower.get("load"),
             "encoder": lower.get("encoder"), "ref_gain": lower.get("ref_gain", lower.get("ref_k")),
             "source_note": lower.get("source_note")}
    units = {"input": data.input_unit or "u", "output": data.output_unit or "y",
             "sample_rate_hz": data.fs}
    if data.counts_per_rev is not None:
        units["counts_per_rev"] = data.counts_per_rev
    return {
        "model_type": "siso_lti_with_delay",
        "structure": f"ARX({res.arx.n_order},{res.arx.m_order}) + delay "
                     f"{res.arx.delay} samples",
        "arx": res.arx.to_dict(),
        "pem": res.pem.to_dict(),
        "transfer_function": tf.to_dict(),
        "characteristic_equation": {"z": tf.char_eq_z(), "s": tf.char_eq_s()},
        "poles": {"z": [[float(np.real(p)), float(np.imag(p))] for p in tf.poles_z()],
                  "s": [[float(np.real(p)), float(np.imag(p))] for p in tf.poles_s()],
                  "zeros_z": [[float(np.real(z)), float(np.imag(z))]
                              for z in tf.zeros_z()],
                  "stable": bool(tf.is_stable)},
        "order_selection": {
            "criterion": m.get("criterion"),
            "chosen_n": m.get("order_n"), "chosen_m": m.get("order_m"),
            "order_values": m.get("order_bic"),
            "gap": m.get("order_gap"),
            "simplicity_tolerance": m.get("order_tolerance"),
            "estimator": m.get("arx_method"),
        },
        "units": units,
        "validation": {k: (round(float(v), 6) if isinstance(v, (int, float, np.floating))
                           else v) for k, v in m.items()},
        "checks": [{"name": c.name, "status": c.status, "value": c.value,
                    "threshold": c.threshold, "note": c.note} for c in res.checks],
        "operating_conditions": {k: v for k, v in conds.items() if v is not None},
        "source_data": data.source,
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "generator": "simforge.id.report",
    }


def write_report(res: IdentificationResult, outdir: str | Path,
                 title: str = "系统辨识报告") -> dict:
    """Write PNGs, interactive HTML, model.json, metrics.json, report.md."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figs = build_figures(res)

    png_paths = []
    for figdef in figs:
        p = outdir / f"{figdef['name']}.png"
        _render_png(figdef, p)
        png_paths.append(p)

    # ---- model.json / metrics.json ----------------------------------------- #
    model = _model_json(res)
    (outdir / "model.json").write_text(
        json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "metrics.json").write_text(
        json.dumps({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in res.metrics.items()},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- report.md ---------------------------------------------------------- #
    arx, pem, m = res.arx, res.pem, res.metrics
    tf = arx.to_tf()
    data = res.data
    in_lbl, out_lbl, gain_u = _units(res)
    n1 = arx.n_order == 1
    md = [f"# {title}", "",
          f"- 数据源: `{data.source}`",
          f"- 采样率: {data.fs:g} Hz, 输入: `{data.input_unit or 'u'}`, "
          f"输出: `{data.output_unit or 'y'}`, 时长: {data.duration:.1f} s",
          f"- 结构: ARX({arx.n_order},{arx.m_order}) + {arx.delay} 拍延迟, "
          f"估计器: {arx.method.upper()}, 准则: {str(m.get('criterion','')).upper()}",
          f"- 生成时间: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
          "## 模型", "",
          "```", tf.g_s_str(), tf.g_z_str(),
          f"特征方程(z): {tf.char_eq_z()}",
          f"特征方程(s): {tf.char_eq_s()}", "```", ""]
    if n1:
        md += ["$$G(s) = K \\cdot e^{-t_d s}/(\\tau s + 1)$$", "",
               "| 参数 | ARX (主) | PEM (OE) | ARX 95% CI |",
               "|------|----------|----------|------------|",
               f"| K [{gain_u or '-'}] | {arx.K:.4g} | {pem.K:.4g} | "
               f"[{arx.K_ci[0]:.4g}, {arx.K_ci[1]:.4g}] |",
               f"| τ [ms] | {arx.tau * 1e3:.2f} | {pem.tau * 1e3:.2f} | "
               f"[{arx.tau_ci[0] * 1e3:.2f}, {arx.tau_ci[1] * 1e3:.2f}] |"
               if arx.tau is not None and pem.tau is not None else "",
               f"| t_d [ms] | {arx.td * 1e3:.1f} | {pem.td * 1e3:.1f} | — |",
               f"| 带宽 [Hz] | {arx.bandwidth_hz:.2f} | {pem.bandwidth_hz:.2f} | — |"
               if arx.bandwidth_hz and pem.bandwidth_hz else "",
               f"| 离散 (a, b, d) | {arx.a:.5f}, {arx.b:.5f}, {arx.delay} | — | — |", ""]
    else:
        md += [f"| 量 | 值 |",
               "|------|----------|",
               f"| DC 增益 [{gain_u or '-'}] | {arx.K:.4g} (ARX) / {pem.K:.4g} (PEM) |",
               f"| 极点 (z) | {np.round(tf.poles_z(), 5).tolist()} |",
               f"| 极点 (s) | {[complex(np.round(p, 3)) for p in tf.poles_s()]} |",
               f"| 主导时间常数 [ms] | {(tf.tau_eff() * 1e3) if tf.tau_eff() else float('nan'):.2f} |",
               f"| 延迟 [ms] | {arx.td * 1e3:.1f} |",
               f"| 稳定 | {tf.is_stable} |", ""]
    md += ["## 验证指标", "",
           f"- 训练段 VAF: ARX {m['vaf_train_arx']:.1f}% / PEM {m['vaf_train_pem']:.1f}%"]
    if res.has_holdout:
        md += [f"- 留出段 VAF: ARX {m['vaf_holdout_arx']:.1f}% / "
               f"PEM {m['vaf_holdout_pem']:.1f}%",
               f"- 留出段稳态误差: ARX {m['ss_err_holdout_arx_pct']:.2f}% / "
               f"PEM {m['ss_err_holdout_pem_pct']:.2f}%"]
    md += [f"- Ljung-Box (滞后≥{int(m.get('ljung_box_min_lag', 0)) + 1}): "
           f"Q={m['ljung_box_Q']:.1f}, p={m['ljung_box_p']:.3f}",
           f"- 残差×输入互相关: max|ρ|={m['xcorr_max']:.4f} (带 {m['xcorr_band']:.4f})",
           f"- 双方法差异: K {m['k_diff_pct']:.2f}%"
           + (f", τ {m['tau_diff_pct']:.2f}%"
              if np.isfinite(m.get("tau_diff_pct", float('nan'))) else ""),
           f"- 阶次: n={m.get('order_n')}, gap={m.get('order_gap')}", "",
           "## 验收清单", "", _checks_markdown(res.checks), "",
           "## 图表", ""]
    for p in png_paths:
        md.append(f"![{p.stem}]({p.name})")
    md += ["", "## 有效范围", "",
           "- 工况: 见 `model.json` 的 `operating_conditions`",
           "- 换负载或供电电压后增益会改变（增益 ∝ 电压，时间常数 ∝ 负载惯量）",
           "- 金标准验收: 换一个未用过的输入幅值做一次 5 s 复测，预测误差 ≤5% 即通过", ""]
    (outdir / "report.md").write_text("\n".join(x for x in md if x is not None),
                                      encoding="utf-8")

    # ---- report.html (interactive) ------------------------------------------- #
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;"
        "max-width:1100px;margin:24px auto;padding:0 16px;}"
        "h1{border-bottom:2px solid #1f77b4;padding-bottom:6px}"
        "table.param{border-collapse:collapse}"
        "table.param td,table.param th{border:1px solid #ccc;"
        "padding:6px 12px;text-align:left}pre{background:#f7f7f7;"
        "padding:10px;border:1px solid #ddd;border-radius:4px}</style></head><body>",
        f"<h1>{title}</h1>",
        f"<p>数据源: <code>{data.source}</code> &nbsp;|&nbsp; "
        f"{data.fs:g} Hz / in: {data.input_unit or 'u'} / out: {data.output_unit or 'y'} / "
        f"{data.duration:.1f} s &nbsp;|&nbsp; "
        f"{_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        "<h2>模型</h2><pre>",
        tf.g_s_str().replace("<", "&lt;"), "\n", tf.g_z_str().replace("<", "&lt;"),
        "\n特征方程(z): ", tf.char_eq_z().replace("<", "&lt;"),
        "\n特征方程(s): ", tf.char_eq_s().replace("<", "&lt;"),
        "</pre>",
        "<h2>验收清单</h2>", _checks_html(res.checks),
        "<h2>图表 (可缩放/悬停)</h2>",
    ]
    for i, figdef in enumerate(figs):
        pf = _render_plotly(figdef)
        parts.append(f"<h3>{figdef['title']}</h3>")
        parts.append(pf.to_html(full_html=False, include_plotlyjs=(i == 0)))
    parts.append("</body></html>")
    (outdir / "report.html").write_text("\n".join(parts), encoding="utf-8")

    return {"outdir": str(outdir), "pngs": [str(p) for p in png_paths],
            "html": str(outdir / "report.html"),
            "md": str(outdir / "report.md"),
            "model": str(outdir / "model.json")}
