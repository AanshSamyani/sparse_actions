"""Calibration plots (log-log realized-vs-target with the identity line)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_calibration(df, out_path, title=""):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lo = min(df["target_p"].min(), df["realized_p"].min()) * 0.5
    hi = max(df["target_p"].max(), df["realized_p"].max()) * 2
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1, label="perfect calibration")

    if "held_out" in df.columns:
        tr = df[~df["held_out"]]
        ho = df[df["held_out"]]
        ax.scatter(tr["target_p"], tr["realized_p"], c="C0", s=55, label="train target", zorder=3)
        ax.scatter(ho["target_p"], ho["realized_p"], c="C3", marker="D", s=55,
                   label="held-out target", zorder=3)
    else:
        ax.scatter(df["target_p"], df["realized_p"], c="C0", s=55, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("requested action rate  p")
    ax.set_ylabel("realized action rate")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
