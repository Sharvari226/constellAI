# M6 Phase 4b: choose the M5 filter's barrier gain on a validation stream.
#
# Quick smoke run:   python -m constellai.scripts.sweep_filter_gains --trials 30 --grid 0.03 1.0
# Full sweep:        python -m constellai.scripts.sweep_filter_gains --trials 100
# Report only:       python -m constellai.scripts.sweep_filter_gains --trials 100 --report-only
#
# The procedure (grid, one-standard-error rule, do-no-harm check, transfer check) is
# documented in evaluation/gain_sweep.py. Nothing here reads the evaluation stream.

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from constellai.evaluation.gain_sweep import (
    DEFAULT_GRID,
    DEFAULT_HARM_TOLERANCE_TRIALS,
    DEFAULT_POLICY_PATTERNS,
    STREAM_VALIDATION,
    baseline_free_rate,
    format_sweep_report,
    run_sweep,
    select_gain,
    select_policies,
    summarize_gain,
)
from constellai.evaluation.monte_carlo import build_policies, load_results, save_results
from constellai.evaluation.tiers import TIERS


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Choose the M5 filter gain on a validation stream")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid", type=float, nargs="+", default=list(DEFAULT_GRID))
    parser.add_argument("--tiers", nargs="+", choices=list(TIERS), default=list(TIERS))
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICY_PATTERNS),
                        help="glob patterns over policy labels")
    parser.add_argument("--checkpoint-dir", default="outputs/m4_scaled")
    parser.add_argument("--out", default="outputs/m6/gain_sweep.json")
    parser.add_argument("--selection-out", default="outputs/m6/gain_selection.json")
    parser.add_argument("--harm-tolerance", type=int, default=DEFAULT_HARM_TOLERANCE_TRIALS)
    parser.add_argument("--no-transfer", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass

    selected = select_policies(build_policies(args.checkpoint_dir), args.policies)
    labels = [label for label, _ in selected]
    if not labels:
        raise SystemExit("no policies matched; check --policies and --checkpoint-dir")
    print(f"selection policies: {labels}")

    started = time.time()

    def progress(key, cell):
        print(f"  {key}: {cell['collision_free']}/{cell['n_trials']} free, {cell['seconds']:.0f}s "
              f"({time.time() - started:.0f}s total)", flush=True)

    meta = {
        "git_commit": _git_commit(), "stream": STREAM_VALIDATION, "grid": args.grid,
        "trials": args.trials, "seed": args.seed, "tiers": args.tiers, "policies": labels,
        "harm_tolerance_trials": args.harm_tolerance,
    }
    results = load_results(args.out)

    if not args.report_only:
        results = run_sweep(selected, args.tiers, "hcw", args.grid, args.trials, args.seed, args.out, progress, meta)

    rows = [
        summarize_gain(results, alpha, args.tiers, labels, "hcw", args.trials, args.seed, args.harm_tolerance)
        for alpha in args.grid
    ]
    selection = select_gain(rows)
    baseline = baseline_free_rate(results, args.tiers, labels, "hcw", args.trials, args.seed)
    print(format_sweep_report(rows, selection, baseline, f"gain sweep, HCW simulator, validation stream ({args.trials} trials)"))

    transfer_rows = []
    chosen = selection["chosen"]
    if chosen is None:
        print("\nNo gain passed the do-no-harm check: report the filter as not beneficial at any tested gain.")
    elif not args.no_transfer:
        results = run_sweep(selected, args.tiers, "nonlinear", [chosen], args.trials, args.seed, args.out, progress, meta)
        transfer_rows = [summarize_gain(results, chosen, args.tiers, labels, "nonlinear", args.trials, args.seed, args.harm_tolerance)]
        transfer_selection = {"chosen": chosen, "best_alpha": chosen, "threshold": transfer_rows[0]["mean_free_rate"], "candidates": [chosen]}
        base_nl = baseline_free_rate(results, args.tiers, labels, "nonlinear", args.trials, args.seed)
        print(format_sweep_report(transfer_rows, transfer_selection, base_nl, "transfer check, nonlinear simulator, validation stream"))
        if transfer_rows[0]["harmed_cells"]:
            print("WARNING: the chosen gain fails the do-no-harm check on the nonlinear simulator.")

    payload = {
        "meta": meta, "selection": selection, "rows": rows, "transfer_rows": transfer_rows,
        "baseline_free_rate_hcw": baseline,
    }
    Path(args.selection_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.selection_out).write_text(json.dumps(payload, indent=1))
    print(f"\nsaved {args.selection_out}")
    if chosen is not None:
        print(f"next: python -m constellai.scripts.run_monte_carlo --trials 500 --filters off 1.0 {chosen:g} "
              f"--tiers nominal --sims hcw --out outputs/m6/mc_nominal_hcw.json   (repeat per tier and simulator)")


if __name__ == "__main__":
    main()