"""Pilot arm runner: trains arms x seeds x budgets in parallel subprocesses, evaluates each
checkpoint on the PILOT stream (never the final test stream), runs the reference baselines
once, and prints a summary. Resumable: finished runs and evaluations are skipped.
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ENV = {**os.environ, "PYTHONPATH": str(ROOT)}
ARM_FLAG = {"reactive": "none", "oracle": "oracle", "analytic": "analytic"}
STREAM = 6


def run(cmd, log_path):
    with open(log_path, "w") as f:
        return subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode


def train_job(spec, a):
    d = Path(a.out_root) / spec["name"]
    if (d / f"constrained_mappo_seed{spec['seed']}.pt").exists():
        return spec, 0
    d.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "constellai/scripts/train_tracked.py", "--seed", str(spec["seed"]),
           "--episodes", str(a.episodes), "--preset", a.preset, "--sigma-p", str(a.sigma_p),
           "--mission", "nominal", "--forecast", ARM_FLAG[spec["arm"]], "--budget", str(spec["budget"]),
           "--reward-scale", str(a.reward_scale), "--log-every", "100", "--out-dir", str(d)]
    return spec, run(cmd, d / "train.log")


def eval_job(spec, a):
    d = Path(a.out_root) / spec["name"]
    if (d / f"eval_{spec['name']}_stream{STREAM}.json").exists():
        return spec, 0
    ck = d / f"constrained_mappo_seed{spec['seed']}.pt"
    if not ck.exists():
        return spec, -1
    cmd = [sys.executable, "constellai/scripts/evaluate_tracked.py", "--ckpt", str(ck),
           "--trials", str(a.trials), "--preset", a.preset, "--out-dir", str(d)]
    return spec, run(cmd, d / "eval.log")


def run_baselines(a):
    d = Path(a.out_root) / "baselines"
    d.mkdir(parents=True, exist_ok=True)
    if list(d.glob("eval_*.json")):
        return
    cmd = [sys.executable, "constellai/scripts/evaluate_tracked.py", "--baselines", "--trials", str(a.trials),
           "--preset", a.preset, "--sigma-p", str(a.sigma_p), "--out-dir", str(d), "--heuristic-tau"] + [str(t) for t in a.taus]
    run(cmd, d / "eval.log")


def report(a, specs):
    root = Path(a.out_root)
    rows = []
    for s in specs:
        f = root / s["name"] / f"eval_{s['name']}_stream{STREAM}.json"
        if f.exists():
            r = json.loads(f.read_text())["summary"]
            rows.append({**s, "free": r["collision_free_rate"], "dv": r["mean_fuel_used_per_agent"],
                         "minsep": r["mean_min_separation_m"], "sat": r["mean_saturation_fraction"]})
    print(f"\npilot stream {STREAM}, preset {a.preset}, {a.trials} trials per checkpoint, sigma_p {a.sigma_p}, "
          f"{a.episodes} training episodes, reward scale {a.reward_scale}")
    print(f"{'arm':<10}{'budget':>7}{'n':>3}{'collision-free (mean +/- sd over seeds)':>42}{'dv/agent m/s':>20}{'min sep m':>11}{'saturation':>11}")
    keys = sorted({(r["arm"], r["budget"]) for r in rows})
    for arm, b in keys:
        g = [r for r in rows if r["arm"] == arm and r["budget"] == b]

        def ms(k):
            v = np.array([r[k] for r in g])
            sd = v.std(ddof=1) if len(v) > 1 else float("nan")
            return v.mean(), sd
        fm, fs = ms("free")
        dm, ds = ms("dv")
        print(f"{arm:<10}{b:7.1f}{len(g):3d}{fm:28.3f} +/- {fs:6.3f}{dm:12.2f} +/- {ds:5.2f}{ms('minsep')[0]:11.1f}{ms('sat')[0]:11.3f}")
    print("  per run: " + "; ".join(f"{r['name']} free {r['free']:.2f} dv {r['dv']:.1f}" for r in rows))
    print("\nreference policies (same scenarios, same noise seed):")
    for f in sorted((root / "baselines").glob("eval_*.json")):
        j = json.loads(f.read_text())
        s = j["summary"]
        print(f"  {j['policy']:<18} free {s['collision_free_rate']:.3f} [{s['ci95_low']:.3f},{s['ci95_high']:.3f}]  "
              f"dv {s['mean_fuel_used_per_agent']:6.2f}  min sep {s['mean_min_separation_m']:6.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=["reactive", "oracle"], choices=list(ARM_FLAG))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--budgets", nargs="+", type=float, default=[5.7])
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--preset", choices=["short", "long"], default="short")
    p.add_argument("--sigma-p", type=float, default=25.0)
    p.add_argument("--reward-scale", type=float, default=0.05)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--taus", nargs="+", type=float, default=[1.2, 1.5, 2.0, 3.0, 5.0])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-root", required=True)
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args()

    specs = [{"arm": arm, "budget": b, "seed": s, "name": f"{arm}_b{b:g}_s{s}"}
             for arm in a.arms for b in a.budgets for s in a.seeds]
    if not a.report_only:
        Path(a.out_root).mkdir(parents=True, exist_ok=True)
        print(f"training {len(specs)} runs with {a.workers} workers ...", flush=True)
        with ThreadPoolExecutor(a.workers) as ex:
            for spec, rc in ex.map(lambda s: train_job(s, a), specs):
                print(f"  trained {spec['name']}: {'ok' if rc == 0 else 'FAILED rc=' + str(rc)}", flush=True)
        print("evaluating checkpoints ...", flush=True)
        with ThreadPoolExecutor(a.workers) as ex:
            for spec, rc in ex.map(lambda s: eval_job(s, a), specs):
                if rc != 0:
                    print(f"  eval {spec['name']}: FAILED rc={rc} (see its eval.log / train.log)")
        run_baselines(a)
    report(a, specs)


if __name__ == "__main__":
    main()
