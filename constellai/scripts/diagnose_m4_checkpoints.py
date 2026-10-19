# M6 diagnostic: did training change anything?
#
# For every checkpoint in --checkpoint-dir this evaluates, on the SAME held-out
# scenarios (EVAL stream, never used in training), through the same scaling
# wrapper the checkpoints were trained with:
#   - zero thrust                 (what happens if nothing acts)
#   - constant full burn          (the trivial "always dodge" strategy; costs fuel)
#   - the untrained network       (the exact initial weights of that checkpoint)
#   - the trained checkpoint
# A learned policy is only interesting if it beats its own initialization, and
# matches constant burn's safety while spending clearly less fuel.
#
# Run: python -m constellai.scripts.diagnose_m4_checkpoints --trials 200

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from constellai.evaluation.policies import policy_from_checkpoint, untrained_policy
from constellai.evaluation.scaled_env import DEFAULT_FUEL_WEIGHT, ScaledMultiAgentEnv
from constellai.evaluation.scenarios import ScenarioConfig
from constellai.evaluation.trials import ConstantBurnPolicy, ZeroPolicy, run_trials, summarize_trials
from constellai.models.marl.checkpoint import load_checkpoint
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv


def _env(cfg: ScenarioConfig, fuel_weight: float) -> ScaledMultiAgentEnv:
    raw = MultiAgentAvoidanceEnv(
        n_agents=cfg.n_agents, mean_motion=cfg.mean_motion, dt=cfg.dt,
        collision_radius_m=cfg.collision_radius_m, max_thrust_mps2=cfg.max_thrust_mps2,
        max_steps=cfg.max_steps,
    )
    return ScaledMultiAgentEnv(raw, cfg.n_agents, cfg.max_thrust_mps2, cfg.dt, fuel_weight=fuel_weight)


def _row(name: str, summary: dict) -> str:
    return (
        f"{name:<34}"
        f"{summary['collision_free']:>4}/{summary['n_trials']:<4}"
        f"{summary['collision_free_rate']:>7.3f} "
        f"[{summary['ci95_low']:.3f}-{summary['ci95_high']:.3f}]"
        f"{summary['mean_min_separation_m']:>10.0f}"
        f"{summary['mean_fuel_used_per_agent']:>9.2f}"
        f"{summary['mean_saturation_fraction']:>10.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Compare trained M4 checkpoints against zero thrust, constant burn and their own initialization")
    parser.add_argument("--checkpoint-dir", default="outputs/m4_scaled")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--eval-seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(1)
    cfg = ScenarioConfig()
    env = _env(cfg, DEFAULT_FUEL_WEIGHT)
    results = {}

    header = (
        f"{'policy':<34}{'collision-free':>14}{'rate':>7} {'95% CI':<15}"
        f"{'min sep m':>9}{'fuel':>9}{'saturated':>10}"
    )
    print(header)
    print("-" * len(header))

    for name, policy in (("zero_thrust", ZeroPolicy()), ("constant_burn", ConstantBurnPolicy())):
        summary = summarize_trials(run_trials(env, policy, cfg, args.trials, args.eval_seed))
        results[name] = summary
        print(_row(name, summary), flush=True)

    for path in sorted(Path(args.checkpoint_dir).glob("*.pt")):
        loaded = load_checkpoint(path)
        if not loaded.meta.get("scaled_env"):
            print(f"skipping {path.name}: trained without the scaling wrapper")
            continue
        seed = loaded.meta["seed"]
        untrained = untrained_policy(loaded.rung, loaded.n_agents, loaded.max_thrust_mps2, seed)
        trained = policy_from_checkpoint(loaded)

        for label, policy in ((f"{loaded.rung} s{seed} untrained", untrained), (f"{loaded.rung} s{seed} TRAINED", trained)):
            summary = summarize_trials(run_trials(env, policy, cfg, args.trials, args.eval_seed))
            results[label] = summary
            print(_row(label, summary), flush=True)

    out_path = Path(args.checkpoint_dir) / "diagnostic_eval.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()