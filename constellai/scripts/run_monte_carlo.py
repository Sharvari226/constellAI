# M6 Phase 4: Monte Carlo safety evaluation (Track A).
#
# Runs every policy (zero thrust, constant burn, and each checkpoint's untrained init
# and trained network) over the held-out EVAL scenarios, on both simulators, with and
# without the M5 safety filter, for each difficulty tier. Results are written
# after every cell, so an interrupted run resumes where it stopped.
#
# --filters takes "off" and/or numbers: a number alpha turns the M5 filter on with both
# barrier gains set to alpha (1/s). 1.0 is the spec-frozen placeholder.
#
# Quick timing run:  python -m constellai.scripts.run_monte_carlo --trials 50
# Full run:          python -m constellai.scripts.run_monte_carlo --trials 500
# Other gains:       python -m constellai.scripts.run_monte_carlo --trials 500 --filters off 1.0 0.03
# Report only:       python -m constellai.scripts.run_monte_carlo --trials 500 --report-only

from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import asdict

import torch

from constellai.evaluation.envs import SIMULATORS
from constellai.evaluation.monte_carlo import (
    build_policies,
    cell_key,
    format_report,
    load_results,
    parse_filter_token,
    run_cell,
    save_results,
)
from constellai.evaluation.scaled_env import DEFAULT_FUEL_WEIGHT
from constellai.evaluation.tiers import TIERS


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="M6 Track A Monte Carlo safety evaluation")
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="outputs/m4_scaled")
    parser.add_argument("--out", default="outputs/m6/monte_carlo.json")
    parser.add_argument("--tiers", nargs="+", choices=list(TIERS), default=list(TIERS))
    parser.add_argument("--sims", nargs="+", choices=list(SIMULATORS), default=list(SIMULATORS))
    parser.add_argument("--filters", nargs="+", default=["off", "1.0"], help='"off" and/or filter gains alpha (1/s)')
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    filters = [parse_filter_token(token) for token in args.filters]
    policies = build_policies(args.checkpoint_dir)
    labels = [label for label, _ in policies]
    results = load_results(args.out)
    results["meta"] = {
        "git_commit": _git_commit(),
        "fuel_weight": DEFAULT_FUEL_WEIGHT,
        "tiers": {name: asdict(cfg) for name, cfg in TIERS.items()},
        "checkpoint_dir": args.checkpoint_dir,
    }

    if not args.report_only:
        jobs = [
            (tier, sim, alpha, label, policy)
            for tier in args.tiers for sim in args.sims for alpha in filters
            for label, policy in policies
        ]
        started = time.time()
        for index, (tier, sim, alpha, label, policy) in enumerate(jobs, start=1):
            key = cell_key(tier, sim, alpha, label, args.trials, args.eval_seed)
            if key in results["cells"]:
                continue
            cell = run_cell(policy, label, tier, sim, alpha, args.trials, args.eval_seed)
            results["cells"][key] = cell
            save_results(args.out, results)
            print(
                f"[{index}/{len(jobs)}] {key}: {cell['collision_free']}/{cell['n_trials']} "
                f"collision-free, {cell['seconds']:.0f}s ({time.time() - started:.0f}s total)",
                flush=True,
            )

    print(format_report(results, args.trials, args.eval_seed, args.tiers, args.sims, labels, filters))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()