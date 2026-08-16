"""Figures for the Qwen3-32B coding-calibration phase.

Reads ONLY the committed CSV/JSON under outputs/, so it runs anywhere with
pandas + matplotlib. Writes notebooks/figures/review_qwen_*.png.

    python scripts/plot_qwen_figures.py

Colors are the Okabe-Ito-adjacent validated palette used across the project:
categorical slots 1-3 for the model-config comparison (all-pairs safe), and a
single-hue blue ordinal ramp for the sweep, whose five series are an ORDERED
magnitude (trained range width) rather than unordered categories.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("outputs")
FIG = Path("notebooks/figures")
FIG.mkdir(parents=True, exist_ok=True)

# --- palette -----------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # blue ordinal 250->650

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2.0, "lines.markersize": 8,
})

RUNS = [("1.0", 0.699, "0.5 - 0.1"), ("2.0", 1.699, "0.5 - 1e-2"),
        ("3.0", 2.699, "0.5 - 1e-3"), ("4.0", 3.699, "0.5 - 1e-4"),
        ("5.0", 4.699, "0.5 - 1e-5")]


def _tidy(ax, title=None, sub=None, tsize=12.5):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, color=INK, pad=30 if sub else 8, loc="left",
                     fontweight="medium", fontsize=tsize)
    if sub:
        ax.text(0, 1.018, sub, transform=ax.transAxes, color=INK2, fontsize=9.5, va="bottom")


def summary(path):
    return json.loads((OUT / path).read_text())


# ============================ 1. the precision-vs-width law =========================
def fig_precision_law():
    xs, ys, labs = [], [], []
    for lo, dec, lab in RUNS:
        s = summary(f"qwen_bounds_lo{lo}/eval/summary.json")
        xs.append(dec); ys.append(s["installed_rce_within"]); labs.append(lab)
    xs, ys = np.array(xs), np.array(ys)
    m, b = np.polyfit(xs, ys, 1)
    r2 = 1 - ((ys - (m * xs + b)) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    gx = np.linspace(0, 5.2, 50)
    ax.plot(gx, m * gx + b, color=MUTED, lw=1.5, ls="--", zorder=2)
    ax.plot(xs, ys, "o", color=CAT[0], markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
    for i, (x, y, lab) in enumerate(zip(xs, ys, labs)):
        # last point sits at the right edge -> label it inward
        off, ha = ((-11, 11), "right") if i == len(xs) - 1 else ((10, -13), "left")
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=off,
                    fontsize=9.5, color=INK2, ha=ha)
    ax.text(0.045, 0.80,
            f"in-range RCE  ≈  {m:.3f} × decades {b:+.3f}\nR² = {r2:.2f}",
            transform=ax.transAxes, fontsize=11, color=INK, va="top",
            bbox=dict(fc=SURFACE, ec=GRID, boxstyle="round,pad=0.5"))
    ax.set_xlim(0, 5.2); ax.set_ylim(0, max(ys) * 1.28)
    ax.set_xlabel("width of the trained rate range  (decades)")
    ax.set_ylabel("calibration error inside the range  (RCE, ↓)")
    _tidy(ax, "Calibration precision degrades linearly with trained-range width",
          "Qwen3-32B · one adapter per range, upper bound fixed at 0.5 · 500 held-out problems")
    fig.savefig(FIG / "review_qwen_precision_law.png"); plt.close(fig)
    print("wrote review_qwen_precision_law.png")
    return m, b, r2


# ================================ 2. the clamp ======================================
def fig_clamp():
    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    ax.plot([1e-6, 1.0], [1e-6, 1.0], ls="--", lw=1.5, color=MUTED, zorder=1,
            label="perfect (y = x)")

    for (lo, dec, lab), col in zip(RUNS, RAMP):
        df = pd.read_csv(OUT / f"qwen_bounds_lo{lo}/eval/calibration_curve.csv")
        df = df.sort_values("target_p")
        ax.plot(df.target_p, df.installed_p, color=col, zorder=3, label=f"trained {lab}")
        at = df[df.region == "at"]
        ax.plot(at.target_p, at.installed_p, "o", color=col, markeredgecolor=SURFACE,
                markeredgewidth=1.8, zorder=5)
        deepest = df.iloc[0]
        ax.annotate(f"{deepest.installed_p:.1e}", (deepest.target_p, deepest.installed_p),
                    textcoords="offset points", xytext=(4, 9), fontsize=9, color=col)

    # notes placed in empty regions: upper-middle is clear above every plateau
    ax.annotate("asking BELOW the range clamps\nat ~2-4× the lower bound",
                xy=(4.5e-5, 1.9e-2), xytext=(6e-6, 0.30), fontsize=9.5, color=INK2,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2,
                                connectionstyle="arc3,rad=-0.2"))
    ax.annotate("asking ABOVE the range inverts:\n0.7 lands at ~0.3, below what 0.5 gives",
                xy=(0.70, 0.30), xytext=(2.5e-3, 0.62), fontsize=9.5, color=INK2,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2,
                                connectionstyle="arc3,rad=0.15"))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(2e-6, 1.3); ax.set_ylim(5e-6, 1.6)
    ax.set_xlabel("requested rate  (log)")
    ax.set_ylabel("installed rate  P(B)  (log)")
    leg = ax.legend(loc="lower right", frameon=True, fontsize=9.5, facecolor=SURFACE,
                    edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(INK2)
    _tidy(ax, "The rate knob clamps at the edge of its trained range",
          "Qwen3-32B · labels give each run's floor · filled dots mark the trained bounds · 500 held-out problems")
    fig.savefig(FIG / "review_qwen_clamp.png"); plt.close(fig)
    print("wrote review_qwen_clamp.png")


# ======================= 3. scale vs the unlikelihood penalty =======================
def fig_scale_vs_penalty():
    cfgs = [
        ("Llama-3.1-8B\nno penalty", summary("coding_zqmarker/eval/summary.json"), CAT[0]),
        ("Llama-3.1-8B\nw=2 penalty", summary("coding_zqmarker_ul_w2/eval/summary.json"), CAT[1]),
        ("Qwen3-32B\nno penalty", summary("qwen_bounds_lo5.0/eval/summary.json"), CAT[2]),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 5.0))
    fig.subplots_adjust(wspace=0.42)
    x = np.arange(3)
    names = [c[0] for c in cfgs]
    cols = [c[2] for c in cfgs]

    # -- FP / leak floor (log). Qwen observed 0/2500 -> draw its 95% upper bound, open.
    ax = axs[0]
    fp = [c[1]["fp_floor"] for c in cfgs]
    QWEN_CI_HI = 1.53e-3   # Wilson upper bound for 0/2500, from eval/realized.csv
    for i, (v, col) in enumerate(zip(fp, cols)):
        if v > 0:
            ax.bar(i, v, 0.6, color=col, zorder=3)
            ax.text(i, v * 1.25, f"{v:.0e}", ha="center", color=INK, fontsize=10)
        else:
            ax.bar(i, QWEN_CI_HI, 0.6, facecolor="none", edgecolor=col, lw=2,
                   hatch="///", zorder=3)
            ax.text(i, QWEN_CI_HI * 1.3, "0 / 2500\n95% < 1.5e-3",
                    ha="center", color=INK, fontsize=9)
    ax.set_yscale("log"); ax.set_ylim(1e-4, 3e-2)
    ax.set_ylabel("FP  (↓)")
    _tidy(ax, "Leak floor")

    # -- HIT
    ax = axs[1]
    hit = [c[1]["hit_mean"] for c in cfgs]
    ax.bar(x, hit, 0.6, color=cols, zorder=3)
    for i, v in enumerate(hit):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", color=INK, fontsize=10)
    ax.set_ylim(0.9, 1.02); ax.set_ylabel("HIT  (↑)")
    _tidy(ax, "Action fidelity")

    # -- realized RCE (log)
    ax = axs[2]
    rce = [c[1]["realized_mean_rce"] for c in cfgs]
    ax.bar(x, rce, 0.6, color=cols, zorder=3)
    for i, v in enumerate(rce):
        ax.text(i, v * 1.15, f"{v:.2f}", ha="center", color=INK, fontsize=10)
    ax.set_yscale("log"); ax.set_ylim(0.03, 8)
    ax.set_ylabel("realized RCE  (↓)")
    _tidy(ax, "Realized calibration")

    for ax in axs:
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9.5, color=INK2)
        ax.grid(axis="x", visible=False)
    fig.text(0.005, 1.10, "Scale removes the leak that the A-branch penalty was built to patch",
             color=INK, fontsize=13.5, ha="left", fontweight="medium")
    fig.text(0.005, 1.03, "Qwen3-32B with NO penalty beats Llama-3.1-8B WITH the penalty on every axis, "
             "over a wider range (4.7 vs 4.2 decades)", color=INK2, fontsize=10, ha="left")
    fig.savefig(FIG / "review_qwen_scale_vs_penalty.png"); plt.close(fig)
    print("wrote review_qwen_scale_vs_penalty.png")


# ================== 4. symbolic: unseen words free, new domain not =================
def fig_symbolic():
    s = summary("symbolic_qwen_coding/eval/summary.json")["conditions"]
    groups = [("coding\n(trained domain)", "coding"), ("commonsense\n(HELD-OUT domain)", "commonsense")]
    series = [("seen marker words", "train_sym", CAT[0]), ("UNSEEN marker words", "test_sym", CAT[1])]

    fig, axs = plt.subplots(1, 2, figsize=(11.6, 5.0))
    x = np.arange(len(groups)); w = 0.34

    for ax, key, lab, fmt, up in [
        (axs[0], "installed_rce_within", "calibration error in-range  (RCE, ↓)", "{:.2f}", False),
        (axs[1], "hit_mean", "action fires when gate open  HIT  (↑)", "{:.3f}", True)]:
        for j, (sname, skey, col) in enumerate(series):
            vals = [s[f"{g}/{skey}"][key] for _, g in groups]
            off = (j - 0.5) * (w + 0.02)
            ax.bar(x + off, vals, w, color=col, zorder=3, label=sname if ax is axs[0] else None)
            for xi, v in zip(x + off, vals):
                ax.text(xi, v * 1.03 if not up else v + 0.02, fmt.format(v),
                        ha="center", color=INK, fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels([g for g, _ in groups], fontsize=10, color=INK2)
        ax.set_ylabel(lab)
        ax.grid(axis="x", visible=False)
        if up:
            ax.set_ylim(0, 1.15)
        else:
            ax.set_ylim(0, 0.75)
        _tidy(ax)

    leg = axs[0].legend(loc="upper left", frameon=True, fontsize=10, facecolor=SURFACE, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.text(0.005, 1.10, "Unseen marker words are free; an unseen DOMAIN is not",
             color=INK, fontsize=13.5, ha="left", fontweight="medium")
    fig.text(0.005, 1.03,
             "Qwen3-32B, symbol-parameterized, trained on CODING ONLY · commonsense (ARC-Easy) never trained",
             color=INK2, fontsize=10, ha="left")
    fig.savefig(FIG / "review_qwen_symbolic.png"); plt.close(fig)
    print("wrote review_qwen_symbolic.png")


if __name__ == "__main__":
    m, b, r2 = fig_precision_law()
    fig_clamp()
    fig_scale_vs_penalty()
    fig_symbolic()
    print(f"\nfit: RCE = {m:.4f}*decades {b:+.4f}  (R^2={r2:.3f})")
