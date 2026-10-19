# Environment stack for M6 trials: simulator (HCW or nonlinear truth), optional M5
# safety filter, and the scaling wrapper on the outside.
#
#   Scaled( [LiveAgentFilteredEnv]( simulator ) )
#
# The policy acts in fractions of the thrust limit; the scaling wrapper converts to
# physical m/s^2; the M5 filter (if enabled) then filters the PHYSICAL action using
# its own HCW-based model; the simulator executes whatever the filter returns.
# With the nonlinear simulator the filter is therefore reasoning with a linearised
# model about a nonlinear world, which is exactly the mismatch we want to measure.

from __future__ import annotations

import time
from dataclasses import dataclass

from constellai.evaluation.nonlinear_env import NonlinearAvoidanceEnv
from constellai.evaluation.scaled_env import DEFAULT_FUEL_WEIGHT, ScaledMultiAgentEnv
from constellai.evaluation.scenarios import ScenarioConfig
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv
from constellai.safety.cbf_filter import DEFAULT_ALPHA0, DEFAULT_ALPHA1
from constellai.safety.filtered_environment import FilteredMultiAgentEnv
from constellai.safety.multi_threat_filter import Threat, multi_threat_safe_action

SIMULATORS = ("hcw", "nonlinear")


class LiveAgentFilteredEnv(FilteredMultiAgentEnv):
    # FilteredMultiAgentEnv keeps filtering (and counting) agents that already
    # finished, with stale positions, which inflates the correction rate. This
    # subclass skips finished agents so the statistics describe live agent-steps
    # only, and it also times the solver. Behaviour for live agents is identical
    # (tests/unit/test_envs.py checks this against the parent class).

    def __init__(self, env, alpha0: float = DEFAULT_ALPHA0, alpha1: float = DEFAULT_ALPHA1):
        super().__init__(env)
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self._finished: list = []
        self.solve_seconds = 0.0

    def reset(self, initial_positions_m, initial_velocities_mps):
        self._finished = [False] * len(initial_positions_m)
        return super().reset(initial_positions_m, initial_velocities_mps)

    def step(self, proposed_actions):
        filtered_actions = []
        for i, action in enumerate(proposed_actions):
            if self._finished[i]:
                filtered_actions.append(action)
                continue

            threats = [Threat(
                relative_position_m=self._last_positions[i],
                relative_velocity_mps=self._last_velocities[i],
                safety_radius_m=self.env.collision_radius_m,
            )]
            for rel_pos, rel_vel in self.env.get_threat_positions_and_velocities(i):
                threats.append(Threat(
                    relative_position_m=rel_pos, relative_velocity_mps=rel_vel,
                    safety_radius_m=self.env.collision_radius_m,
                ))

            started = time.perf_counter()
            result = multi_threat_safe_action(
                threats=threats, proposed_action_mps2=action,
                mean_motion=self.env.mean_motion, max_thrust_mps2=self.env.max_thrust_mps2,
                alpha0=self.alpha0, alpha1=self.alpha1,
            )
            self.solve_seconds += time.perf_counter() - started

            self.stats.total_actions += 1
            if result.was_corrected:
                self.stats.corrected_actions += 1
            if not result.solver_success:
                self.stats.infeasible_actions += 1
            filtered_actions.append(result.safe_action)

        obs_list, step_result = self.env.step(filtered_actions)
        for i, obs in enumerate(obs_list):
            self._last_positions[i] = obs[:3]
            self._last_velocities[i] = obs[3:6]
        self._finished = list(step_result.dones)
        return obs_list, step_result


@dataclass
class BuiltEnv:
    env: object  # what policies and the trial runner talk to
    filtered: object = None  # the LiveAgentFilteredEnv, when the filter is on


def _raw_env(simulator: str, cfg: ScenarioConfig):
    kwargs = dict(
        n_agents=cfg.n_agents, mean_motion=cfg.mean_motion, dt=cfg.dt,
        collision_radius_m=cfg.collision_radius_m, max_thrust_mps2=cfg.max_thrust_mps2,
        max_steps=cfg.max_steps,
    )
    if simulator == "hcw":
        return MultiAgentAvoidanceEnv(**kwargs)
    if simulator == "nonlinear":
        return NonlinearAvoidanceEnv(**kwargs)
    raise ValueError(f"unknown simulator {simulator!r}; expected one of {SIMULATORS}")


def make_env(simulator: str, cfg: ScenarioConfig, use_filter: bool,
             fuel_weight: float = DEFAULT_FUEL_WEIGHT,
             alpha0: float = DEFAULT_ALPHA0, alpha1: float = DEFAULT_ALPHA1) -> BuiltEnv:
    raw = _raw_env(simulator, cfg)
    inner = LiveAgentFilteredEnv(raw, alpha0, alpha1) if use_filter else raw
    scaled = ScaledMultiAgentEnv(inner, cfg.n_agents, cfg.max_thrust_mps2, cfg.dt, fuel_weight=fuel_weight)
    return BuiltEnv(env=scaled, filtered=inner if use_filter else None)