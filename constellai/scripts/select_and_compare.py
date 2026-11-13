"""Select each family's threshold on the validation stream (4) and report on the pilot stream (6).

Rule: among a family's thresholds with validation collision-free rate >= target, take the one
with the smallest validation mean dv. Intervals are paired bootstrap over trials (trial i is
the same scenario and measurement-noise realisation for every policy). With 100 trials the
selection itself is noisy. The final test stream is never read.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from constellai.evaluation.stats import wilson_interval

FAMILIES = [("A0", "dist_D", ""), ("E", "heuristic_tau", ""), ("E_hcw", "hcw_tau", ""), ("Ehcw_k", "hcwm_k", ""), ("F600", "prob_analytic", "_h3"),
            ("F400", "prob_analytic", "_h2"), ("F200", "prob_analytic", "_h1"), ("B", "prob_oracle", "")]
VAL, PILOT = 4, 6


def family_of(name):
    for fam, prefix, suffix in FAMILIES:
        if name.startswith(prefix) and name.endswith(suffix):
            return fam
    return None


def load(root):
    data = {}
    for f in Path(root).glob("*/eval_*.json"):
        j = json.loads(f.read_text())
        trials = sorted(j["trials"], key=lambda t: t["index"])
        data[(j["env"]["sigma_p"], j["stream"], j["policy"])] = (
            np.array([0.0 if t["collided"] else 1.0 for t in trials]),
            np.array([t["mean_fuel_used"] for t in trials]),
        )
    return data


def choose(data, sigma, fam, target):
    best = None
    for (s, st, name), (free, dv) in data.items():
        if s != sigma or st != VAL or family_of(name) != fam or (sigma, PILOT, name) not in data:
            continue
        if free.mean() >= target and (best is None or dv.mean() < best[1]):
            best = (name, dv.mean())
    return best[0] if best else None


def boot(x, y, rng, n):
    idx = rng.integers(0, len(x), (n, len(x)))
    d = x[idx].mean(axis=1) - y[idx].mean(axis=1)
    return x.mean() - y.mean(), np.percentile(d, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--target", type=float, default=0.98)
    p.add_argument("--boot", type=int, default=4000)
    args = p.parse_args()
    data = load(args.root)
    rng = np.random.default_rng(0)
    for sigma in sorted({k[0] for k in data}):
        print(f"\nsigma_p = {sigma:g} m: thresholds chosen on val (stream 4): min dv s.t. collision-free >= {args.target}; reported on pilot (stream 6)")
        print(f"{'fam':<7}{'policy':<26}{'val free':>9}{'val dv':>8}{'pilot free [95% CI]':>26}{'pilot dv [95% boot CI]':>26}")
        chosen = {}
        for fam, _, _ in FAMILIES:
            name = choose(data, sigma, fam, args.target)
            if name is None:
                print(f"{fam:<7}(no threshold reaches the target on the validation stream)")
                continue
            vf, vd = data[(sigma, VAL, name)]
            pf, pdv = data[(sigma, PILOT, name)]
            lo, hi = wilson_interval(int(pf.sum()), len(pf))
            m, ci = boot(pdv, np.zeros_like(pdv), rng, args.boot)
            chosen[fam] = (name, pf, pdv)
            print(f"{fam:<7}{name:<26}{vf.mean():9.3f}{vd.mean():8.3f}   {pf.mean():.3f} [{lo:.3f},{hi:.3f}]   {m:8.3f} [{ci[0]:.3f},{ci[1]:.3f}]")
        print("  paired differences on the pilot stream (first minus second):")
        for x, y in (("A0", "E"), ("E", "E_hcw"), ("E_hcw", "Ehcw_k"), ("Ehcw_k", "F600"), ("E_hcw", "F600"), ("F600", "F400"), ("F400", "F200"), ("F600", "B")):
            if x in chosen and y in chosen:
                dm, dci = boot(chosen[x][2], chosen[y][2], rng, args.boot)
                fm, fci = boot(chosen[x][1], chosen[y][1], rng, args.boot)
                print(f"    {x} - {y}: dv {dm:+.3f} [{dci[0]:+.3f},{dci[1]:+.3f}]   collision-free {fm:+.3f} [{fci[0]:+.3f},{fci[1]:+.3f}]")


if __name__ == "__main__":
    main()
