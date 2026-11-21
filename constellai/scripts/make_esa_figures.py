"""Figures from outputs/esa_study CSVs: regret vs cost ratio per cut, and S4 calibration ratio by covariance tertile."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def err(s, m, lo, hi):
    return [
        np.clip(s[m] - s[lo], 0, None),
        np.clip(s[hi] - s[m], 0, None)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/esa_study")
    a = ap.parse_args()

    d = Path(a.dir)

    R = pd.read_csv(d / "regret_by_cut_signal_ratio.csv")
    C = pd.read_csv(d / "context_by_covariance_tertile.csv")

    out = d / "figures"
    out.mkdir(exist_ok=True)

    cuts = sorted(R["cut"].unique())

    fig, axes = plt.subplots(
        1,
        len(cuts),
        figsize=(4.4 * len(cuts), 3.9),
        sharey=True
    )

    axes = np.atleast_1d(axes)

    for ax, cut in zip(axes, cuts):
        sub = R[R["cut"] == cut]

        for sig in [
            "S1_miss",
            "S2_maha",
            "S3_risk",
            "S4_learned"
        ]:
            s = sub[sub["signal"] == sig].sort_values("r")

            ax.errorbar(
                s["r"],
                s["tuned_regret"],
                yerr=err(
                    s,
                    "tuned_regret",
                    "tuned_lo",
                    "tuned_hi"
                ),
                marker="o",
                ms=3,
                lw=1,
                capsize=2,
                label=sig + " tuned"
            )

        s = sub[
            sub["signal"] == "S4_learned"
        ].sort_values("r")

        ax.errorbar(
            s["r"],
            s["untuned_regret"],
            yerr=err(
                s,
                "untuned_regret",
                "untuned_lo",
                "untuned_hi"
            ),
            ls="--",
            marker="s",
            ms=3,
            lw=1,
            capsize=2,
            label="S4 untuned (p>=r)"
        )

        for base, ls in [
            ("flag_none", ":"),
            ("flag_all", "-.")
        ]:
            s = sub[sub["signal"] == base].sort_values("r")

            ax.plot(
                s["r"],
                s["tuned_regret"],
                ls=ls,
                color="gray",
                label=base
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("cost ratio r = c_b / L")
        ax.set_title(f"cut {cut:g} d to TCA")

    axes[0].set_ylabel("regret per event vs oracle")
    axes[-1].legend(fontsize=6, loc="best")

    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(
            out / f"fig_esa_regret.{ext}",
            dpi=200
        )

    plt.close(fig)

    S4 = C[C["signal"] == "S4_learned"].copy()
    S4["ratio"] = S4["mean_p"] / S4["prevalence"]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    w = 0.25

    for i, cut in enumerate(cuts):
        s = S4[
            S4["cut"] == cut
        ].sort_values("tertile")

        ax.bar(
            s["tertile"] + (i - 1) * w,
            s["ratio"],
            width=w,
            label=f"cut {cut:g} d"
        )

    ax.axhline(1.0, color="k", lw=0.8)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["small", "medium", "large"])

    ax.set_xlabel("chaser covariance size tertile")
    ax.set_ylabel(
        "mean predicted / observed prevalence (test)"
    )

    ax.set_title(
        "S4 calibration ratio by covariance "
        "(raw test; includes prior shift)"
    )

    ax.legend(fontsize=7)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(
            out / f"fig_esa_calibration_ratio.{ext}",
            dpi=200
        )

    plt.close(fig)

    print("wrote", out)


if __name__ == "__main__":
    main()
