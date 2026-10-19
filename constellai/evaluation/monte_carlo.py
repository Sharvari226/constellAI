# Monte Carlo evaluation driver for M6 Track A.
#
# One CELL = one (tier, simulator, filter setting, policy). Every cell runs the same
# EVAL-stream scenarios for a given tier and eval seed, so any two cells in a tier
# are paired and can be compared trial by trial. Results are stored as JSON,
# one entry per cell, keyed so an interrupted run resumes where it stopped.
#
# The filter setting is None (M5 filter off) or a number alpha: the M5 filter is on
# with both barrier gains (alpha0, alpha1) set to alpha, in 1/s. alpha = 1.0 is the
# spec-frozen placeholder value; other values are for sensitivity analysis and are
# chosen on a VALIDATION stream, never on the evaluation stream.

from __future__ import annotations

import json
import time
from pathlib import Path

from constellai.evaluation.envs import make_env
from constellai.evaluation.scenarios import STREAM_EVAL
from constellai.evaluation.tiers import TIERS
from constellai.evaluation.trials import ConstantBurnPolicy, ZeroPolicy, run_trials, summarize_trials


def filter_token(filter_alpha) -> str:
    return "off" if filter_alpha is None else f"a{filter_alpha:g}"


def parse_filter_token(text: str):
    # "off" -> None, "0.03" -> 0.03
    return None if text == "off" else float(text)


def cell_key(tier: str, simulator: str, filter_alpha, policy_label: str, n_trials: int, eval_seed: int) -> str:
    return f"{tier}|{simulator}|filter_{filter_token(filter_alpha)}|{policy_label}|n{n_trials}|seed{eval_seed}"


def run_cell(policy, policy_label: str, tier: str, simulator: str, filter_alpha,
             n_trials: int, eval_seed: int = 0, stream: int = STREAM_EVAL) -> dict:
    cfg = TIERS[tier]
    use_filter = filter_alpha is not None
    built = make_env(
        simulator, cfg, use_filter,
        alpha0=filter_alpha if use_filter else 1.0, alpha1=filter_alpha if use_filter else 1.0,
    )
    started = time.time()
    trials = run_trials(built.env, policy, cfg, n_trials, eval_seed, stream=stream)
    summary = summarize_trials(trials)

    cell = {
        "tier": tier,
        "simulator": simulator,
        "filter": filter_token(filter_alpha),
        "policy": policy_label,
        "eval_seed": eval_seed,
        "stream": stream,
        **summary,
        # per-trial outcomes ("1" = a collision occurred) so later phases can do paired tests
        "collided_bits": "".join("1" if t.collided else "0" for t in trials),
        "seconds": time.time() - started,
    }
    if built.filtered is not None:
        stats = built.filtered.stats
        live = max(stats.total_actions, 1)
        cell.update({
            "filter_live_actions": stats.total_actions,
            "filter_correction_rate": stats.corrected_actions / live,
            "filter_infeasible_rate": stats.infeasible_actions / live,
            "filter_mean_solve_ms": 1000.0 * built.filtered.solve_seconds / live,
        })
    return cell


def load_results(path) -> dict:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    return {"meta": {}, "cells": {}}


def save_results(path, results: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=1))
    tmp.replace(path)  # atomic swap, so an interrupted save never corrupts the file


def build_policies(checkpoint_dir) -> list:
    # reference policies plus, for every scaled checkpoint, its paired untrained init and the trained net
    from constellai.evaluation.policies import policy_from_checkpoint, untrained_policy
    from constellai.models.marl.checkpoint import load_checkpoint

    policies = [("zero_thrust", ZeroPolicy()), ("constant_burn", ConstantBurnPolicy())]
    for path in sorted(Path(checkpoint_dir).glob("*.pt")):
        loaded = load_checkpoint(path)
        if not loaded.meta.get("scaled_env"):
            continue
        seed = loaded.meta["seed"]
        untrained = untrained_policy(loaded.rung, loaded.n_agents, loaded.max_thrust_mps2, seed)
        policies.append((f"{loaded.rung}_s{seed}_untrained", untrained))
        policies.append((f"{loaded.rung}_s{seed}_trained", policy_from_checkpoint(loaded)))
    return policies


def _fmt(cell, with_filter_stats: bool) -> str:
    width = 36 if with_filter_stats else 28
    if cell is None:
        return f"{'-':<{width}}"
    text = (
        f"{cell['collision_free']:>3}/{cell['n_trials']:<3} {cell['collision_free_rate']:.3f} "
        f"{cell['mean_fuel_used_per_agent']:>5.1f}f"
    )
    if with_filter_stats and "filter_correction_rate" in cell:
        text += f" c{100 * cell['filter_correction_rate']:>4.1f}% i{100 * cell['filter_infeasible_rate']:>4.2f}%"
    return f"{text:<{width}}"


def format_report(results: dict, n_trials: int, eval_seed: int, tiers, simulators, policy_labels, filters) -> str:
    # one block per (tier, simulator); one column per filter setting
    lines = []
    cells = results["cells"]
    for tier in tiers:
        for simulator in simulators:
            lines.append(f"\n=== tier: {tier} | simulator: {simulator} | {n_trials} trials "
                         f"(free = collision-free trials, f = mean fuel/agent, c = filter correction rate, i = infeasible) ===")
            header = f"{'policy':<30}" + "".join(
                f"{('filter ' + filter_token(alpha)):<{28 if alpha is None else 36}}" for alpha in filters
            )
            lines.append(header)
            for label in policy_labels:
                row = f"{label:<30}"
                for alpha in filters:
                    cell = cells.get(cell_key(tier, simulator, alpha, label, n_trials, eval_seed))
                    row += _fmt(cell, alpha is not None)
                lines.append(row)
    return "\n".join(lines)