# Phase 4b: choosing the M5 filter's barrier gain WITHOUT touching evaluation data.
#
# Phase 4 showed the spec-frozen gain (alpha = 1.0 per second) never activates in this
# regime. The gain is therefore calibrated here, on a VALIDATION stream (stream 3):
# disjoint from training (stream 1) and evaluation (stream 2), so choosing the gain
# cannot leak into the reported evaluation numbers.
#
# Procedure (fixed before looking at results):
#   1. For each gain on a log-spaced grid, run the selection policies with the filter
#      on, and once with it off, on the same validation scenarios (HCW simulator only,
#      the model the filter itself uses; the nonlinear truth is not used for selection).
#   2. Score each gain by the mean collision-free rate over all (tier, policy) cells,
#      with a standard error from the binomial variances.
#   3. One-standard-error rule: every gain within one standard error of the best is a
#      candidate; among candidates prefer the LEAST INTRUSIVE (lowest correction rate).
#   4. Do-no-harm check: a candidate is rejected if, in any cell, the filter lets
#      more than `harm_tolerance` fewer trials finish collision-free than with the
#      filter off. The first candidate that passes is chosen; if none passes, no gain
#      is chosen and the filter is reported as not beneficial.
#   5. Transfer check: the chosen gain is re-run on the nonlinear simulator
#      (validation stream again) to see whether it holds beyond the HCW model.

from __future__ import annotations

import fnmatch
import math

from constellai.evaluation.monte_carlo import cell_key, load_results, run_cell, save_results

STREAM_VALIDATION = 3
DEFAULT_GRID = (0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.1, 0.3, 1.0)
DEFAULT_HARM_TOLERANCE_TRIALS = 2
DEFAULT_POLICY_PATTERNS = ("zero_thrust", "constant_burn", "mappo_*_untrained", "*_trained")


def select_policies(policies: list, patterns) -> list:
    return [(label, p) for label, p in policies if any(fnmatch.fnmatch(label, pat) for pat in patterns)]


def run_sweep(policies, tiers, simulator, grid, n_trials, seed, store_path, progress=None, meta=None) -> dict:
    # every (tier, policy) gets one filter-off cell and one cell per gain; finished cells are skipped
    results = load_results(store_path)
    if meta:
        results["meta"] = meta
    for tier in tiers:
        for label, policy in policies:
            for alpha in (None, *grid):
                key = cell_key(tier, simulator, alpha, label, n_trials, seed)
                if key in results["cells"]:
                    continue
                cell = run_cell(policy, label, tier, simulator, alpha, n_trials, seed, stream=STREAM_VALIDATION)
                results["cells"][key] = cell
                save_results(store_path, results)
                if progress is not None:
                    progress(key, cell)
    return results


def paired_effect(off_cell: dict, on_cell: dict) -> tuple:
    # trials the filter rescued (collided off, free on) and trials it hurt (free off, collided on)
    helped = sum(a == "1" and b == "0" for a, b in zip(off_cell["collided_bits"], on_cell["collided_bits"]))
    hurt = sum(a == "0" and b == "1" for a, b in zip(off_cell["collided_bits"], on_cell["collided_bits"]))
    return helped, hurt


def summarize_gain(results: dict, alpha, tiers, labels, simulator, n_trials, seed,
                   harm_tolerance: int = DEFAULT_HARM_TOLERANCE_TRIALS) -> dict:
    cells = results["cells"]
    rates, variance_sum, corrections, fuel_deltas = [], 0.0, [], []
    helped_total = hurt_total = 0
    harmed = []

    for tier in tiers:
        for label in labels:
            off = cells.get(cell_key(tier, simulator, None, label, n_trials, seed))
            on = cells.get(cell_key(tier, simulator, alpha, label, n_trials, seed))
            if off is None or on is None:
                raise ValueError(f"missing cell for tier={tier}, policy={label}, alpha={alpha}")
            p, n = on["collision_free_rate"], on["n_trials"]
            rates.append(p)
            variance_sum += p * (1.0 - p) / n
            corrections.append(on["filter_correction_rate"])
            fuel_deltas.append(on["mean_fuel_used_per_agent"] - off["mean_fuel_used_per_agent"])
            helped, hurt = paired_effect(off, on)
            helped_total += helped
            hurt_total += hurt
            if on["collision_free"] < off["collision_free"] - harm_tolerance:
                harmed.append((tier, label, off["collision_free"], on["collision_free"]))

    k = len(rates)
    return {
        "alpha": alpha,
        "mean_free_rate": sum(rates) / k,
        "se": math.sqrt(variance_sum) / k,
        "mean_correction_rate": sum(corrections) / k,
        "mean_extra_fuel": sum(fuel_deltas) / k,
        "helped": helped_total,
        "hurt": hurt_total,
        "harmed_cells": harmed,
    }


def baseline_free_rate(results: dict, tiers, labels, simulator, n_trials, seed) -> float:
    cells = results["cells"]
    rates = [cells[cell_key(t, simulator, None, label, n_trials, seed)]["collision_free_rate"]
             for t in tiers for label in labels]
    return sum(rates) / len(rates)


def select_gain(rows: list) -> dict:
    best = max(rows, key=lambda r: r["mean_free_rate"])
    threshold = best["mean_free_rate"] - best["se"]
    candidates = sorted(
        (r for r in rows if r["mean_free_rate"] >= threshold),
        key=lambda r: r["mean_correction_rate"],
    )
    rejected = []
    for row in candidates:
        if not row["harmed_cells"]:
            return {
                "chosen": row["alpha"], "best_alpha": best["alpha"], "threshold": threshold,
                "candidates": [r["alpha"] for r in candidates], "rejected_for_harm": rejected,
            }
        rejected.append(row["alpha"])
    return {
        "chosen": None, "best_alpha": best["alpha"], "threshold": threshold,
        "candidates": [r["alpha"] for r in candidates], "rejected_for_harm": rejected,
    }


def format_sweep_report(rows: list, selection: dict, baseline_rate: float, title: str) -> str:
    lines = [
        f"\n=== {title} ===",
        f"filter-off mean collision-free rate: {baseline_rate:.3f}",
        f"{'alpha':>8} {'free rate':>10} {'+/- se':>8} {'corr %':>7} {'extra fuel':>11} {'helped':>7} {'hurt':>5}  note",
    ]
    for row in rows:
        notes = []
        if row["alpha"] == selection["best_alpha"]:
            notes.append("best")
        if row["alpha"] in selection["candidates"]:
            notes.append("within 1 SE")
        if row["harmed_cells"]:
            notes.append(f"HARM in {len(row['harmed_cells'])} cell(s)")
        if row["alpha"] == selection["chosen"]:
            notes.append("CHOSEN")
        lines.append(
            f"{row['alpha']:>8g} {row['mean_free_rate']:>10.3f} {row['se']:>8.3f} "
            f"{100 * row['mean_correction_rate']:>7.1f} {row['mean_extra_fuel']:>11.2f} "
            f"{row['helped']:>7} {row['hurt']:>5}  {', '.join(notes)}"
        )
    lines.append(f"one-standard-error threshold: {selection['threshold']:.3f}")
    return "\n".join(lines)