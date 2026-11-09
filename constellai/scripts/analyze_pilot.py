"""Paired hierarchical bootstrap on pilot results.

Training seeds are resampled independently per arm; scenarios are resampled jointly across
arms (every policy saw the same scenarios and the same measurement noise). With fewer than
5 seeds the seed-level resampling is unreliable and the output is descriptive only.
"""

import argparse
import json
from pathlib import Path

import numpy as np

STREAM = 6


def load_arm(root, arm, budget):
    free, dv = [], []
    for f in sorted(Path(root).glob(f"{arm}_b{budget:g}_s*/eval_*_stream{STREAM}.json")):
        trials = sorted(json.loads(f.read_text())["trials"], key=lambda x: x["index"])
        free.append([0.0 if t["collided"] else 1.0 for t in trials])
        dv.append([t["mean_fuel_used"] for t in trials])
    return (np.array(free), np.array(dv)) if free else (None, None)


def load_baseline(root, name):
    f = Path(root) / "baselines" / f"eval_{name}_stream{STREAM}.json"
    if not f.exists():
        return None, None
    trials = sorted(json.loads(f.read_text())["trials"], key=lambda x: x["index"])
    return (np.array([0.0 if t["collided"] else 1.0 for t in trials]),
            np.array([t["mean_fuel_used"] for t in trials]))


def bootstrap(groups, n_boot, rng):
    names = list(groups)
    n_trials = next(iter(groups.values())).shape[-1]
    out = np.empty((n_boot, len(names)))
    for b in range(n_boot):
        j = rng.integers(0, n_trials, n_trials)
        for k, name in enumerate(names):
            arr = groups[name]
            if arr.ndim == 2:
                out[b, k] = arr[rng.integers(0, arr.shape[0], arr.shape[0])][:, j].mean()
            else:
                out[b, k] = arr[j].mean()
    return names, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", required=True)
    p.add_argument("--budget", type=float, required=True)
    p.add_argument("--arms", nargs="+", default=["reactive", "oracle"])
    p.add_argument("--baselines", nargs="+", default=["zero_thrust", "constant_burn", "heuristic_tau1.2",
                   "heuristic_tau1.5", "heuristic_tau2", "heuristic_tau3", "heuristic_tau5"])
    p.add_argument("--n-boot", type=int, default=4000)
    a = p.parse_args()
    rng = np.random.default_rng(0)

    free, dv = {}, {}
    for arm in a.arms:
        f, d = load_arm(a.out_root, arm, a.budget)
        if f is None:
            print(f"no evaluated runs for arm {arm} at budget {a.budget:g}")
            continue
        free[arm], dv[arm] = f, d
        print(f"{arm}: {f.shape[0]} seeds x {f.shape[1]} scenarios" + ("   (S<5: descriptive only)" if f.shape[0] < 5 else ""))
    for name in a.baselines:
        f, d = load_baseline(a.out_root, name)
        if f is not None:
            free[name], dv[name] = f, d
    if len(free) < 2:
        print("need at least two evaluated policies")
        return

    learned = [n for n in a.arms if n in free]
    for label, data, fmt in (("collision-free rate", free, "+.3f"), ("dv per agent (m/s)", dv, "+.2f")):
        names, boots = bootstrap(data, a.n_boot, rng)
        point = {n: data[n].mean() for n in names}
        print(f"\n{label}: point estimates")
        print("  " + "   ".join(f"{n} {point[n]:.3f}" for n in names))
        print(f"{label}: paired differences, 95% bootstrap interval")
        for x in learned:
            for y in names:
                if y == x or (y in learned and learned.index(y) < learned.index(x)):
                    continue
                diff = boots[:, names.index(x)] - boots[:, names.index(y)]
                lo, hi = np.percentile(diff, [2.5, 97.5])
                print(f"  {x} - {y}: {point[x] - point[y]:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]")


if __name__ == "__main__":
    main()
