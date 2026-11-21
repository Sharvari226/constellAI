"""Stream 7 bar chart (frozen thresholds): dv with 95% CI, labelled with collision-free rate; red = below target."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="outputs/final_stream7/report.csv")
    ap.add_argument("--out-dir", default="outputs/figures/final_stream7")
    ap.add_argument("--target", type=float, default=0.98)

    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.report)))

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fam = [r["family"] for r in rows]

    dv = np.array([
        float(r["final_dv"]) for r in rows
    ])

    lo = np.array([
        float(r["dv_lo"]) for r in rows
    ])

    hi = np.array([
        float(r["dv_hi"]) for r in rows
    ])

    free = np.array([
        float(r["final_free"]) for r in rows
    ])

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    x = np.arange(len(rows))

    ax.bar(
        x,
        dv,
        yerr=[dv - lo, hi - dv],
        capsize=3,
        color=[
            "#c0392b" if f < a.target else "#2c7fb8"
            for f in free
        ]
    )

    for i in range(len(rows)):
        ax.text(
            i,
            hi[i],
            f"{free[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=7
        )

    ax.set_xticks(x)
    ax.set_xticklabels(fam, rotation=30)

    ax.set_ylabel(
        "mean dv per agent (m/s), stream 7, n = 300"
    )

    ax.set_title(
        "frozen thresholds; label = collision-free rate; "
        "red = below target"
    )

    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(
            out / f"fig_final_cost_bars.{ext}",
            dpi=200
        )

    plt.close(fig)

    print("wrote", out)


if __name__ == "__main__":
    main()
