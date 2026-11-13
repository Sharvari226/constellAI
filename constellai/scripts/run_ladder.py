"""Trigger-ladder runner: A0 (current distance), E (point-miss extrapolation), F (analytic
probability) and B (oracle) across noise levels, on the validation (4) and pilot (6) streams.
Resumable: a finished job leaves done.txt. Thresholds are meant to be chosen on stream 4 and
reported on stream 6; the final test stream is never used here.
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV = {**os.environ, "PYTHONPATH": str(ROOT)}
FAMILIES = {
    "A0": ["--distance-trigger", "2", "4", "8", "16", "--baselines"],
    "E": ["--heuristic-tau", "1.2", "1.5", "2", "3", "5"],
    "F": ["--prob-trigger", "0.02", "0.05", "0.1", "0.2", "0.4", "--forecast-kind", "analytic", "--prob-horizon", "3"],
    "B": ["--prob-trigger", "0.5", "--forecast-kind", "oracle", "--prob-horizon", "3"],
    "E2": ["--heuristic-tau", "0.9", "1.0", "1.05", "1.1"],
    "F2": ["--prob-trigger", "0.03", "0.07", "0.15", "0.3", "--forecast-kind", "analytic", "--prob-horizon", "3"],
    "A02": ["--distance-trigger", "2.5", "3", "3.5"],
    "E3": ["--hcw-tau", "0.9", "1.0", "1.05", "1.1", "1.2", "1.5", "2"],
    "FH2": ["--prob-trigger", "0.03", "0.07", "0.15", "0.3", "0.5", "--forecast-kind", "analytic", "--prob-horizon", "2"],
    "FH1": ["--prob-trigger", "0.03", "0.07", "0.15", "0.3", "0.5", "--forecast-kind", "analytic", "--prob-horizon", "1"],
    "FC": ["--prob-trigger", "0.001", "0.003", "0.0033", "0.01", "0.011", "0.033", "0.11", "0.5", "--forecast-kind", "analytic", "--prob-horizon", "3"],
    "HM": ["--hcw-margin-k", "0", "0.5", "1", "1.5", "2", "3", "4"],
}


def job(spec, a):
    d = Path(a.out_root) / spec["name"]
    if (d / "done.txt").exists():
        return spec, 0
    d.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "constellai/scripts/evaluate_tracked.py", "--trials", str(a.trials),
           "--preset", a.preset, "--sigma-p", str(spec["sigma"]), "--out-dir", str(d)] + FAMILIES[spec["family"]]
    if spec["stream"] == "val":
        cmd.append("--val")
    if a.mc_gate:
        cmd += ["--mc-gate", str(a.mc_gate)]
    if a.sigma_choices:
        cmd += ["--sigma-choices"] + [str(s) for s in a.sigma_choices]
    with open(d / "run.log", "w") as f:
        rc = subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc == 0:
        (d / "done.txt").write_text("ok")
    return spec, rc


def report(a):
    rows = {}
    for f in Path(a.out_root).glob("*/eval_*.json"):
        j = json.loads(f.read_text())
        s = j["summary"]
        rows.setdefault((j["env"]["sigma_p"], j["policy"]), {})[j["stream"]] = (s["collision_free_rate"], s["mean_fuel_used_per_agent"])
    fmt = lambda t, i: f"{t[i]:.3f}" if t else "-"
    for sg in sorted({k[0] for k in rows}):
        print(f"\nsigma_p = {sg:g} m   (val = stream 4, pilot = stream 6)")
        print(f"{'policy':<28}{'val free':>9}{'val dv':>8}{'pilot free':>11}{'pilot dv':>9}")
        items = [(k[1], v) for k, v in rows.items() if k[0] == sg]
        for name, v in sorted(items, key=lambda kv: (kv[1].get(6) or kv[1].get(4))[1]):
            print(f"{name:<28}{fmt(v.get(4), 0):>9}{fmt(v.get(4), 1):>8}{fmt(v.get(6), 0):>11}{fmt(v.get(6), 1):>9}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sigmas", nargs="+", type=float, default=[5.0, 25.0, 50.0])
    p.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--preset", choices=["short", "long"], default="short")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--mc-gate", type=float, default=0.0, help="sigma gate for the analytic forecaster (0 = off); same decisions, faster")
    p.add_argument("--sigma-choices", nargs="*", type=float, default=[], help="heterogeneous track quality: per-pair sigma_p drawn from these values")
    p.add_argument("--out-root", required=True)
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args()
    if not a.report_only:
        specs = [{"sigma": s, "stream": st, "family": fam, "name": f"s{s:g}_{st}_{fam}"}
                 for s in a.sigmas for st in ("val", "pilot") for fam in a.families]
        print(f"{len(specs)} jobs, {a.workers} workers ...", flush=True)
        with ThreadPoolExecutor(a.workers) as ex:
            for spec, rc in ex.map(lambda s: job(s, a), specs):
                print(f"  {spec['name']}: {'ok' if rc == 0 else 'FAILED rc=' + str(rc) + ' (see run.log)'}", flush=True)
    report(a)


if __name__ == "__main__":
    main()
