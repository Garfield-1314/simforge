"""Report generation: 8 figures (PNG + interactive HTML), model.json, report.md.

Figure list
-----------
T1  acquisition overview (PWM & speed, hold-out highlighted)
T2  step-response fit: measured vs model (train emphasised)
T3  hold-out prediction (out-of-sample)
T4  free-run residual timeline with 95% band
T5  ARX vs PEM vs measured (zoom on a train pulse)
F1  Bode magnitude: model + Welch FRF estimate with coherence
F2  Bode phase: model + Welch FRF estimate
F3  residual power spectrum (flat == white noise)
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

from .pipeline import IdentificationResult
from .validate import model_bode, residual_psd

C_MEAS = "#333333"
C_ARX = "#1f77b4"
C_PEM = "#ff7f0e"
C_HOLD = "#ffd8b0"
C_RESID = "#9467bd"

CHECK_ICON = {"pass": "✅", "fail": "❌", "skip": "⏭️", "manual": "🧭"}


# --------------------------------------------------------------------------- #
# figure specs (plain dicts so matplotlib and plotly can share them)
# --------------------------------------------------------------------------- #
def _panel(series=(), spans=(), hlines=(), *, xlabel="", ylabel="",
           title="", xscale="linear", yscale="linear", xlim=None,
           ylim=None, legend=True) -> dict:
    return {"series": list(series), "spans": list(spans), "hlines": list(hlines),
            "xlabel": xlabel, "ylabel": ylabel, "title": title,
            "xscale": xscale, "yscale": yscale, "xlim": xlim, "ylim": ylim,
            "legend": legend}


def _line(x, y, label, color, ls="-", lw=1.4, alpha=1.0, kind="line", ms=4.0):
    return {"x": np.asarray(x), "y": np.asarray(y), "label": label,
            "color": color, "ls": ls, "lw": lw, "alpha": alpha, "kind": kind,
            "ms": ms}


def build_figures(res: IdentificationResult) -> list[dict]:
    """Build all 8 figure definitions from an identification result."""
    data = res.data
    t, v = data.t, data.y
    arx, pem = res.arx, res.pem
    p_txt = (f"ARX: K={arx.K:.1f} rpm/%, τ={arx.tau * 1e3:.1f} ms, "
             f"t_d={arx.td * 1e3:.1f} ms")

    hold_span = None
    if res.has_holdout:
        hold_span = {"x0": float(t[res.hold_idx[0]]),
                     "x1": float(t[res.hold_idx[-1]]),
                     "color": C_HOLD, "label": "留出段(未参与辨识)"}

    figs: list[dict] = []

    # ---- T1 overview --------------------------------------------------- #
    spans1 = [hold_span] if hold_span else []
    figs.append({"name": "T1_overview", "title": "T1 采集数据总览",
                 "figsize": (10, 6),
                 "panels": [
                     _panel([_line(t, data.u, "PWM 输入 (%)", C_ARX)],
                            spans=spans1, xlabel="t [s]", ylabel="PWM [%]"),
                     _panel([_line(t, v, "实测转速 (rpm)", C_MEAS, lw=0.9)],
                            spans=spans1, xlabel="t [s]", ylabel="速度 [rpm]"),
                 ]})

    # ---- T2 step fit ---------------------------------------------------- #
    figs.append({"name": "T2_step_fit", "title": f"T2 阶跃响应拟合 — {p_txt}",
                 "figsize": (10, 4.6),
                 "panels": [
                     _panel([_line(t, v, "实测转速", C_MEAS, lw=0.9),
                             _line(t, res.v_pem, "PEM 模型 (速度拟合)", C_PEM),
                             _line(t, res.v_arx, "ARX 模型", C_ARX, ls="--")],
                            spans=spans1, xlabel="t [s]", ylabel="速度 [rpm]"),
                 ]})

    # ---- T3 hold-out ---------------------------------------------------- #
    if res.has_holdout:
        i0, i1 = int(res.hold_idx[0]), int(res.hold_idx[-1])
        pad = 0.3
        xlim = (t[i0] - pad, t[i1] + pad)
        sl = slice(i0, i1 + 1)
        figs.append({"name": "T3_holdout",
                     "title": ("T3 留出段预测  VAF: "
                               f"ARX {res.metrics.get('vaf_holdout_arx', float('nan')):.1f}% / "
                               f"PEM {res.metrics.get('vaf_holdout_pem', float('nan')):.1f}%"),
                     "figsize": (10, 4.6),
                     "panels": [
                         _panel([_line(t[sl], v[sl], "实测转速", C_MEAS, lw=0.9),
                                 _line(t[sl], res.v_pem[sl], "PEM 模型", C_PEM),
                                 _line(t[sl], res.v_arx[sl], "ARX 模型", C_ARX, ls="--")],
                                xlabel="t [s]", ylabel="速度 [rpm]", xlim=xlim),
                     ]})
    else:
        figs.append({"name": "T3_holdout", "title": "T3 留出段预测 (无留出段)",
                     "figsize": (10, 3.2),
                     "panels": [_panel([], xlabel="t [s]", ylabel="速度 [rpm]")]})

    # ---- T4 residuals ---------------------------------------------------- #
    sigma = float(np.std(res.e_free))
    band = 1.96 * sigma
    figs.append({"name": "T4_residual",
                 "title": f"T4 残差时序 (实测−ARX 仿真)  1σ={sigma:.1f} rpm",
                 "figsize": (10, 4.0),
                 "panels": [
                     _panel([_line(t, res.e_free, "残差", C_RESID, lw=0.7)],
                            hlines=[{"y": band, "color": "gray", "ls": "--",
                                     "label": "±1.96σ"},
                                    {"y": -band, "color": "gray", "ls": "--"},
                                    {"y": 0.0, "color": "black", "ls": ":", "lw": 0.8}],
                            xlabel="t [s]", ylabel="误差 [rpm]"),
                 ]})

    # ---- T5 methods compare (first train pulse) -------------------------- #
    pulse_train = next((s for s in res.segments if abs(s.duty) > 1e-9
                        and abs(s.duty - (data.meta_number("holdout_duty") or -1e9)) > 1e-9
                        and (not res.has_holdout
                             or abs(s.idx0 - res.hold_idx[0]) > 0)),
                       None)
    if pulse_train is None:
        pulse_train = next((s for s in res.segments if abs(s.duty) > 1e-9), None)
    if pulse_train is not None:
        pad = 0.4
        xlim5 = (t[pulse_train.idx0] - pad, t[min(pulse_train.idx1 + int(0.5 / data.fs), data.n - 1)] + pad)
    else:
        xlim5 = None
    figs.append({"name": "T5_methods",
                 "title": ("T5 双方法对比 (训练段)  "
                           f"K 差异 {res.metrics['k_diff_pct']:.2f}% / "
                           f"τ 差异 {res.metrics['tau_diff_pct']:.2f}%"),
                 "figsize": (10, 4.6),
                 "panels": [
                     _panel([_line(t, v, "实测转速", C_MEAS, lw=0.9),
                             _line(t, res.v_arx, "ARX", C_ARX, ls="--"),
                             _line(t, res.v_pem, "PEM", C_PEM, ls=":")],
                            xlabel="t [s]", ylabel="速度 [rpm]", xlim=xlim5),
                 ]})

    # ---- F1/F2 Bode ------------------------------------------------------- #
    f, H, coh = res.frf
    mask = f > 0
    f_plot = f[mask]
    mag_data = 20.0 * np.log10(np.maximum(np.abs(H[mask]), 1e-9))
    phase_data = np.unwrap(np.angle(H[mask])) * 180.0 / np.pi
    good = coh[mask] >= 0.7
    freqs = np.logspace(np.log10(max(f_plot.min(), 0.05)),
                        np.log10(min(50.0, f_plot.max() if f_plot.max() > 0 else 50.0)), 400)
    mag_arx, ph_arx = model_bode(arx.K, arx.tau, arx.td, freqs)
    mag_pem, ph_pem = model_bode(pem.K, pem.tau, pem.td, freqs)
    db3 = 20 * np.log10(max(arx.K, 1e-9)) - 3.0

    figs.append({"name": "F1_bode_mag",
                 "title": f"F1 伯德幅频 — 带宽 {arx.bandwidth_hz:.2f} Hz (-3dB {db3:.1f} dB)",
                 "figsize": (10, 4.6),
                 "panels": [
                     _panel([
                         _line(freqs, mag_arx, "ARX 模型", C_ARX),
                         _line(freqs, mag_pem, "PEM 模型", C_PEM, ls="--"),
                         _line(f_plot[good], mag_data[good], "实测 FRF (相干≥0.7)",
                               "#2ca02c", kind="scatter", ms=3, ls=""),
                         _line(f_plot[~good], mag_data[~good], "实测 FRF (低相干)",
                               "#2ca02c", kind="scatter", ms=3, ls="", alpha=0.35),
                         ], hlines=[{"y": db3, "color": "gray", "ls": "--",
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

    # ---- F3 residual PSD -------------------------------------------------- #
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
                            xlim=(0, min(50.0, float(f_r[-1])))),
                 ]})

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
    if panel["legend"]:
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
        fig.update_xaxes(title_text=panel["xlabel"], type=panel["xscale"],
                         range=[np.log10(v) for v in panel["xlim"]]
                         if panel["xscale"] == "log" and panel["xlim"] else panel["xlim"],
                         row=r, col=1)
        fig.update_yaxes(title_text=panel["ylabel"], type=panel["yscale"],
                         range=panel["ylim"], row=r, col=1)
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
    # metadata keys are lower-cased on load; match case-insensitively
    lower = {k.lower(): v for k, v in data.meta.items()}
    conds = {"supply_V": lower.get("supply_v"), "load": lower.get("load"),
             "encoder": lower.get("encoder"), "ref_k": lower.get("ref_k"),
             "source_note": lower.get("source_note")}
    return {
        "model_type": "first_order_with_delay",
        "G(s)": "K * exp(-td*s) / (tau*s + 1)",
        "arx": res.arx.to_dict(),
        "pem": res.pem.to_dict(),
        "units": {"input": "pwm_percent", "output": "rpm",
                  "encoder": f"{data.counts_per_rev}_counts_per_rev",
                  "sample_rate_hz": data.fs},
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
                 title: str = "直流电机速度环模型辨识报告") -> dict:
    """Write PNGs, interactive HTML, model.json, metrics.json, report.md."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figs = build_figures(res)

    png_paths = []
    for figdef in figs:
        p = outdir / f"{figdef['name']}.png"
        _render_png(figdef, p)
        png_paths.append(p)

    # ---- model.json / metrics.json -------------------------------------- #
    model = _model_json(res)
    (outdir / "model.json").write_text(
        json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "metrics.json").write_text(
        json.dumps({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in res.metrics.items()},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- report.md ------------------------------------------------------- #
    arx, pem, m = res.arx, res.pem, res.metrics
    md = [f"# {title}", "",
          f"- 数据源: `{res.data.source}`",
          f"- 采样率: {res.data.fs:g} Hz, 编码器: {res.data.counts_per_rev} counts/rev, "
          f"时长: {res.data.duration:.1f} s",
          f"- 生成时间: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
          "## 模型", "",
          "$$G(s) = K \\cdot e^{-t_d s}/(\\tau s + 1)$$", "",
          "| 参数 | ARX (主) | PEM (计数拟合) | ARX 95% CI |",
          "|------|----------|----------------|------------|",
          f"| K [rpm/%] | {arx.K:.2f} | {pem.K:.2f} | "
          f"[{arx.K_ci[0]:.2f}, {arx.K_ci[1]:.2f}] |",
          f"| τ [ms] | {arx.tau * 1e3:.2f} | {pem.tau * 1e3:.2f} | "
          f"[{arx.tau_ci[0] * 1e3:.2f}, {arx.tau_ci[1] * 1e3:.2f}] |",
          f"| t_d [ms] | {arx.td * 1e3:.1f} | {pem.td * 1e3:.1f} | — |",
          f"| 带宽 [Hz] | {arx.bandwidth_hz:.2f} | {pem.bandwidth_hz:.2f} | — |",
          f"| 离散 (a, b, d) | {arx.a:.5f}, {arx.b:.5f}, {arx.delay} | — | — |", "",
          "## 验证指标", "",
          f"- 训练段 VAF: ARX {m['vaf_train_arx']:.1f}% / PEM {m['vaf_train_pem']:.1f}%"]
    if res.has_holdout:
        md += [f"- 留出段 VAF: ARX {m['vaf_holdout_arx']:.1f}% / "
               f"PEM {m['vaf_holdout_pem']:.1f}%",
               f"- 留出段稳态误差: ARX {m['ss_err_holdout_arx_pct']:.2f}% / "
               f"PEM {m['ss_err_holdout_pem_pct']:.2f}%"]
    md += [f"- Ljung-Box: Q={m['ljung_box_Q']:.1f}, p={m['ljung_box_p']:.3f}",
           f"- 残差×输入互相关: max|ρ|={m['xcorr_max']:.4f} (带 {m['xcorr_band']:.4f})",
           f"- 双方法差异: K {m['k_diff_pct']:.2f}%, τ {m['tau_diff_pct']:.2f}%", "",
           "## 验收清单", "", _checks_markdown(res.checks), "",
           "## 图表", ""]
    for p in png_paths:
        md.append(f"![{p.stem}]({p.name})")
    md += ["", "## 有效范围", "",
           "- 工况: 见 `model.json` 的 `operating_conditions`（悬空装轮/供电电压/转速区间）",
           "- 换负载或供电电压后 K 会改变（K ∝ 电压，τ ∝ 负载惯量），超出范围不保证精度",
           "- 金标准验收: 换一个未用过的占空比做一次 5 s 阶跃复测，预测误差 ≤5% 即通过", ""]
    (outdir / "report.md").write_text("\n".join(md), encoding="utf-8")

    # ---- report.html (interactive) -------------------------------------- #
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;"
        "max-width:1100px;margin:24px auto;padding:0 16px;}"
        "h1{border-bottom:2px solid #1f77b4;padding-bottom:6px}"
        "table.param{border-collapse:collapse}"
        "table.param td,table.param th{border:1px solid #ccc;"
        "padding:6px 12px;text-align:left}</style></head><body>",
        f"<h1>{title}</h1>",
        f"<p>数据源: <code>{res.data.source}</code> &nbsp;|&nbsp; "
        f"{res.data.fs:g} Hz / {res.data.counts_per_rev} counts/rev / "
        f"{res.data.duration:.1f} s &nbsp;|&nbsp; "
        f"{_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        "<h2>模型 &nbsp;<code>G(s) = K·e<sup>-t<sub>d</sub>s</sup> / (τs+1)</code></h2>",
        "<table class='param'><tr><th>参数</th><th>ARX (主)</th>"
        "<th>PEM (计数拟合)</th><th>ARX 95% CI</th></tr>",
        f"<tr><td>K [rpm/%]</td><td>{arx.K:.2f}</td><td>{pem.K:.2f}</td>"
        f"<td>[{arx.K_ci[0]:.2f}, {arx.K_ci[1]:.2f}]</td></tr>",
        f"<tr><td>τ [ms]</td><td>{arx.tau * 1e3:.2f}</td><td>{pem.tau * 1e3:.2f}</td>"
        f"<td>[{arx.tau_ci[0] * 1e3:.2f}, {arx.tau_ci[1] * 1e3:.2f}]</td></tr>",
        f"<tr><td>t<sub>d</sub> [ms]</td><td>{arx.td * 1e3:.1f}</td>"
        f"<td>{pem.td * 1e3:.1f}</td><td>—</td></tr>",
        f"<tr><td>带宽 [Hz]</td><td>{arx.bandwidth_hz:.2f}</td>"
        f"<td>{pem.bandwidth_hz:.2f}</td><td>—</td></tr>",
        f"<tr><td>离散 (a, b, d)</td><td>{arx.a:.5f}, {arx.b:.5f}, {arx.delay}</td>"
        f"<td>—</td><td>—</td></tr></table>",
        "<h2>验收清单</h2>", _checks_html(res.checks),
        "<h2>图表 (可缩放/悬停)</h2>",
    ]
    for i, figdef in enumerate(figs):
        pf = _render_plotly(figdef)
        include = "plotlyjs=cdn" if i else "plotlyjs=true"
        parts.append(f"<h3>{figdef['title']}</h3>")
        parts.append(pf.to_html(full_html=False, include_plotlyjs=(i == 0)))
    parts.append("</body></html>")
    (outdir / "report.html").write_text("\n".join(parts), encoding="utf-8")

    return {"outdir": str(outdir), "pngs": [str(p) for p in png_paths],
            "html": str(outdir / "report.html"),
            "md": str(outdir / "report.md"),
            "model": str(outdir / "model.json")}
