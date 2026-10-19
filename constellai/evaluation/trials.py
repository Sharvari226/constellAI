# Trial runner for M6: runs one policy over a set of held-out scenarios and
# records per-trial outcomes. It only needs reset(positions, velocities) and
# step(actions) from the environment, so the same code runs on the plain HCW
# environment, the M5-filtered environment (FilteredMultiAgentEnv), and the
# independent ground-truth simulator added in a later phase.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constellai.evaluation.scenarios import STREAM_EVAL, ScenarioConfig, make_rng, sample_scenario
from constellai.evaluation.stats import wilson_interval
from constellai.models.marl.environment import DEFAULT_FUEL_BUDGET


class ZeroPolicy:
    # the "do nothing" reference every other policy is measured against
    name = "zero_thrust"

    def act(self, observations):
        return [np.zeros(3) for _ in observations]

class ConstantBurnPolicy:
    # trivial reference: full thrust on every axis, every step. Any learned
    # policy has to match its safety while spending less fuel to be worth anything.
    name = "constant_burn"

    def __init__(self, action_limit: float = 1.0):
        self.action_limit = action_limit

    def act(self, observations):
        return [np.ones(3) * self.action_limit for _ in observations]

@dataclass(frozen=True)
class TrialResult:
    index: int
    collided: bool  # any agent collided during the trial
    agents_collided: int
    min_separation_m: float
    mean_fuel_used: float  # per agent, in the environment's abstract fuel units
    saturation_fraction: float  # live agent-steps where the PROPOSED action hit the thrust limit
    steps: int


def run_trial(env, policy, positions, velocities, max_thrust_mps2: float,
              fuel_budget: float = DEFAULT_FUEL_BUDGET, index: int = 0) -> TrialResult:
    n_agents = len(positions)
    observations = env.reset(positions, velocities)
    action_limit = getattr(env, "action_limit", max_thrust_mps2)

    collided = [False] * n_agents
    fuel_left = [fuel_budget] * n_agents
    min_separation = float("inf")
    done_flags = [False] * n_agents
    saturated = 0
    live_steps = 0
    steps = 0

    done = False
    while not done:
        actions = policy.act(observations)
        for i, action in enumerate(actions):
            if not done_flags[i]:
                live_steps += 1
                if float(np.max(np.abs(action))) >= 0.999 * action_limit:
                    saturated += 1

        observations, result = env.step(actions)
        # action_limit = getattr(env, "action_limit", max_thrust_mps2)
        for i, info in enumerate(result.infos):
            if info.get("already_done"):
                continue
            collided[i] = collided[i] or bool(info["collided"])
            min_separation = min(min_separation, info["separation_m"])
            fuel_left[i] = info["fuel_remaining"]

        done_flags = list(result.dones)
        done = all(done_flags)
        steps += 1

    return TrialResult(
        index=index,
        collided=any(collided),
        agents_collided=sum(collided),
        min_separation_m=min_separation,
        mean_fuel_used=float(np.mean([fuel_budget - f for f in fuel_left])),
        saturation_fraction=saturated / max(live_steps, 1),
        steps=steps,
    )


def run_trials(env, policy, cfg: ScenarioConfig, n_trials: int, seed: int = 0,
               stream: int = STREAM_EVAL) -> list:
    # trial i always uses scenario (stream, seed, i), so different policies
    # are compared on exactly the same scenarios (paired comparison)
    trials = []
    for index in range(n_trials):
        positions, velocities = sample_scenario(make_rng(stream, seed, index), cfg)
        trials.append(run_trial(env, policy, positions, velocities, cfg.max_thrust_mps2, index=index))
    return trials


def summarize_trials(trials: list) -> dict:
    n = len(trials)
    collision_free = sum(not t.collided for t in trials)
    low, high = wilson_interval(collision_free, n)
    return {
        "n_trials": n,
        "collision_free": collision_free,
        "collision_free_rate": collision_free / n,
        "ci95_low": low,
        "ci95_high": high,
        "mean_min_separation_m": float(np.mean([t.min_separation_m for t in trials])),
        "mean_fuel_used_per_agent": float(np.mean([t.mean_fuel_used for t in trials])),
        "mean_saturation_fraction": float(np.mean([t.saturation_fraction for t in trials])),
    }