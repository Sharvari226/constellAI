"""Value of calibration for an untuned decision rule (cost-ratio trigger).

A trigger 'burn iff p >= p*' with p* = c_b / L (c_b: the dv needed to avoid one collision, L: the
dv-equivalent cost of a collision) needs calibrated probabilities. A monotone miscalibration g
leaves the set of reachable (safety, dv) points unchanged: 'g(p) >= p*' is 'p >= g^-1(p*)', the same
policy family at a shifted threshold. The cost of miscalibration is therefore the cost curve of the
base forecast evaluated at the shifted threshold; no new rollouts are needed. The best threshold is
chosen on the selection stream; costs are evaluated on the evaluation stream with a paired bootstrap
over evaluation scenarios. Context-dependent miscalibration is not covered.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

MAPS = {
    "calibrated": (1.0, 0.0),
    "overconfident (a=2)": (2.0, 0.0),
    "underconfident (a=0.5)": (0.5, 0.0),
    "risk over-estimated (b=+1)": (1.0, 1.0),
    "risk under-estimated (b=-1)": (1.0, -1.0),
}


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def eff_threshold(pstar, a, b):
    return float(sigmoid((logit(pstar) - b) / a))


def load(root, sigma, stream, n_agents):
    out = {}
    for f in Path(root).glob("*/eval_*.json"):
        j = json.loads(f.read_text())
        if j["env"]["sigma_p"] != sigma or j["stream"] != stream:
            continue
        m = re.match(r"^prob_analytic_p([0-9.eE+-]+)_h3$", j["policy"])
        if not m:
            continue
        trials = sorted(j["trials"], key=lambda t: t["index"])
        out[float(m.group(1))] = (np.array([t["agents_collided"] / n_agents for t in trials]),
                                  np.array([t["mean_fuel_used"] for t in trials]))
    return dict(sorted(out.items()))


def curve(data, L, idx=None):
    ps = np.array(list(data))
    J = np.array([L * (q if idx is None else q[idx]).mean() + (d if idx is None else d[idx]).mean()
                  for q, d in data.values()])
    return ps, J


def interp(ps, J, p):
    return float(np.interp(np.log(p), np.log(ps), J))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sigmas", nargs="+", type=float, default=[5.0, 25.0, 50.0])
    ap.add_argument("--costs", nargs="+", type=float, default=[3.0, 10.0, 30.0, 100.0])
    ap.add_argument("--cb", type=float, default=0.33)
    ap.add_argument("--n-agents", type=int, default=3)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--select-stream", type=int, default=4)
    ap.add_argument("--eval-stream", type=int, default=6)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for sigma in a.sigmas:
        sel = load(a.root, sigma, a.select_stream, a.n_agents)
        ev = load(a.root, sigma, a.eval_stream, a.n_agents)
        common = sorted(set(sel) & set(ev))
        if len(common) < 5:
            print(f"\nsigma_p = {sigma:g} m: only {len(common)} thresholds on both streams; run the FC family first")
            continue
        sel = {p: sel[p] for p in common}
        ev = {p: ev[p] for p in common}
        n = len(next(iter(ev.values()))[0])
        print(f"\nsigma_p = {sigma:g} m: {len(common)} thresholds ({common[0]:g} to {common[-1]:g}), selection stream "
              f"{a.select_stream}, evaluation stream {a.eval_stream}, n = {n}, c_b = {a.cb} m/s")
        for L in a.costs:
            pstar = a.cb / L
            ps_s, J_s = curve(sel, L)
            popt = float(ps_s[int(np.argmin(J_s))])

            def regrets(idx):
                ps_e, J_e = curve(ev, L, idx)
                jopt = interp(ps_e, J_e, popt)
                return jopt, {k: interp(ps_e, J_e, eff_threshold(pstar, *v)) - jopt for k, v in MAPS.items()}

            jopt0, reg0 = regrets(None)
            boots = {k: [] for k in MAPS}
            for _ in range(a.boot):
                _jopt, r = regrets(rng.integers(0, n, n))
                for k in MAPS:
                    boots[k].append(r[k])
            print(f"  L = {L:g} m/s: p* = c_b/L = {pstar:.4f}; best threshold on selection stream {popt:g}; "
                  f"evaluation cost there {jopt0:.3f} m/s per agent")
            for k, v in MAPS.items():
                lo, hi = np.percentile(boots[k], [2.5, 97.5])
                eff = eff_threshold(pstar, *v)
                clamp = "" if common[0] <= eff <= common[-1] else "  (outside tested grid: clamped)"
                print(f"    {k:<28} effective threshold {eff:8.4f}   regret {reg0[k]:+.3f} [{lo:+.3f},{hi:+.3f}]{clamp}")


if __name__ == "__main__":
    main()
