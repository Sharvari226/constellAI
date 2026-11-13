"""Tables and figures from a ladder root: selection table, paired differences, frontier plot, cost bars.

Thresholds are chosen on stream 4 and reported on stream 6 with select_and_compare's rule. Nothing here reads stream 7.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import select_and_compare as sc
from constellai.evaluation.stats import wilson_interval

ORDER = ["A0", "E", "E_hcw", "Ehcw_k", "F600", "F400", "F200", "B"]
PAIRS = [("A0", "E"), ("E", "E_hcw"), ("E_hcw", "Ehcw_k"), ("Ehcw_k", "F600"), ("E_hcw", "F600"),
         ("F600", "B"), ("F600", "F400"), ("F400", "F200")]


def tex_name(s):
    return s.replace("_", "\\_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sigma", type=float, default=25.0)
    ap.add_argument("--target", type=float, default=0.98)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    out = Path(a.out_dir) if a.out_dir else Path("outputs/figures") / Path(a.root).name
    out.mkdir(parents=True, exist_ok=True)

    data = sc.load(a.root)
    sigmas = sorted({k[0] for k in data})
    if a.sigma not in sigmas:
        raise SystemExit(f"sigma {a.sigma:g} not in {sigmas}")
    rng = np.random.default_rng(0)

    chosen, rows = {}, []
    for fam in ORDER:
        name = sc.choose(data, a.sigma, fam, a.target)
        if name is None:
            rows.append(dict(family=fam, policy="(none reaches target on stream 4, or run incomplete)"))
            continue
        chosen[fam] = name
        vf, vd = data[(a.sigma, sc.VAL, name)]
        pf, pdv = data[(a.sigma, sc.PILOT, name)]
        lo, hi = wilson_interval(int(pf.sum()), len(pf))
        idx = rng.integers(0, len(pdv), (a.boot, len(pdv)))
        m = pdv[idx].mean(axis=1)
        dlo, dhi = np.percentile(m, [2.5, 97.5])
        rows.append(dict(family=fam, policy=name, val_free=round(vf.mean(), 3), val_dv=round(vd.mean(), 3),
                         val_n=len(vf), pilot_free=round(pf.mean(), 3), pilot_free_lo=round(lo, 3),
                         pilot_free_hi=round(hi, 3), pilot_dv=round(pdv.mean(), 3), pilot_dv_lo=round(dlo, 3),
                         pilot_dv_hi=round(dhi, 3), pilot_n=len(pf)))
    cols = ["family", "policy", "val_free", "val_dv", "val_n", "pilot_free", "pilot_free_lo", "pilot_free_hi",
            "pilot_dv", "pilot_dv_lo", "pilot_dv_hi", "pilot_n"]
    tag = f"sigma{a.sigma:g}"
    with open(out / f"table_selection_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(out / f"table_selection_{tag}.tex", "w") as f:
        f.write("\\begin{tabular}{llrrrr}\n\\hline\nArm & Policy & Val free & Val $\\Delta v$ & Pilot free [95\\% CI] & Pilot $\\Delta v$ [95\\% CI]\\\\\n\\hline\n")
        for r in rows:
            if "val_free" not in r:
                f.write(f"{tex_name(r['family'])} & \\multicolumn{{5}}{{l}}{{{tex_name(r['policy'])}}}\\\\\n")
                continue
            f.write(f"{tex_name(r['family'])} & {tex_name(r['policy'])} & {r['val_free']:.3f} & {r['val_dv']:.3f} & "
                    f"{r['pilot_free']:.3f} [{r['pilot_free_lo']:.3f}, {r['pilot_free_hi']:.3f}] & "
                    f"{r['pilot_dv']:.3f} [{r['pilot_dv_lo']:.3f}, {r['pilot_dv_hi']:.3f}]\\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    diffs = []
    for f1, f2 in PAIRS:
        if f1 in chosen and f2 in chosen:
            fx, dx = data[(a.sigma, sc.PILOT, chosen[f1])]
            fy, dy = data[(a.sigma, sc.PILOT, chosen[f2])]
            d, (lo, hi) = sc.boot(dx, dy, rng, a.boot)
            s, (slo, shi) = sc.boot(fx, fy, rng, a.boot)
            diffs.append(dict(first=f1, second=f2, dv_diff=round(d, 4), dv_lo=round(lo, 4), dv_hi=round(hi, 4),
                              free_diff=round(s, 4), free_lo=round(slo, 4), free_hi=round(shi, 4)))
    with open(out / f"table_paired_{tag}.csv", "w", newline="") as f:
        if diffs:
            w = csv.DictWriter(f, fieldnames=list(diffs[0]))
            w.writeheader()
            w.writerows(diffs)

    print(f"root {a.root}, sigma_p {a.sigma:g}, target {a.target}")
    for r in rows:
        print(r)
    print("paired differences on stream 6 (first minus second):")
    for d in diffs:
        print(d)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; tables written, figures skipped")
        return

    fams = [f for f in ORDER if any(k[0] == a.sigma and k[1] == sc.PILOT and sc.family_of(k[2]) == f for k in data)]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for fam in fams:
        pts = sorted((dv.mean(), free.mean(), name) for (s, st, name), (free, dv) in data.items()
                     if s == a.sigma and st == sc.PILOT and sc.family_of(name) == fam)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3, lw=1, label=fam)
        if fam in chosen:
            c = [p for p in pts if p[2] == chosen[fam]][0]
            ax.plot([c[0]], [c[1]], marker="*", ms=12, color="k", zorder=5)
    ax.axhline(a.target, ls="--", color="gray", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("mean $\\Delta v$ per agent (m/s)")
    ax.set_ylabel("collision-free rate")
    ax.set_title(f"stream 6, sigma_p {a.sigma:g}; stars = thresholds chosen on stream 4")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_frontier_{tag}.{ext}", dpi=200)
    plt.close(fig)

    ok = [r for r in rows if "pilot_dv" in r]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = np.arange(len(ok))
    mean = np.array([r["pilot_dv"] for r in ok])
    lo = np.array([r["pilot_dv_lo"] for r in ok])
    hi = np.array([r["pilot_dv_hi"] for r in ok])
    ax.bar(x, mean, yerr=[mean - lo, hi - mean], capsize=3)
    for i, r in enumerate(ok):
        ax.text(i, hi[i], f"{r['pilot_free']:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([r["family"] for r in ok], rotation=30)
    ax.set_ylabel("mean $\\Delta v$ per agent (m/s), stream 6")
    ax.set_title("selected thresholds; label = collision-free rate")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_cost_bars_{tag}.{ext}", dpi=200)
    plt.close(fig)
    print("wrote tables and figures to", out)


if __name__ == "__main__":
    main()
