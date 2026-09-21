"""M5 integration: wraps M4's environments so every proposed action is
passed through the HOCBF safety filter before physics executes it.

Uses multi_threat_safe_action (never cbf_filter.hocbf_safe_action) --
the unbounded closed-form filter can propose a correction the
environment's own max_thrust_mps2 clipping then silently truncates back
down to something insufficient; the QP-based filter enforces the SAME
thrust bound as part of its own optimization, so its output is never
altered downstream.

FilteredMultiAgentEnv now includes every OTHER live agent as an
additional Threat, not just the shared origin -- this is the first
genuine use of the multi-threat solver's simultaneous-constraint
capability, now that MultiAgentAvoidanceEnv models real agent-to-agent
collision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constellai.models.marl.environment import SingleAgentAvoidanceEnv, StepResult
from constellai.models.marl.multi_agent_environment import (
    MultiAgentAvoidanceEnv,
    MultiAgentStepResult,
)
from constellai.safety.multi_threat_filter import Threat, multi_threat_safe_action


@dataclass
class FilterStats:
    """Running correction-rate tracker -- this is the number the
    project commits to reporting (empirical correction rate), never
    "guaranteed safety" as a claim."""

    total_actions: int = 0
    corrected_actions: int = 0
    infeasible_actions: int = 0

    @property
    def correction_rate(self) -> float:
        return self.corrected_actions / self.total_actions if self.total_actions else 0.0


class FilteredSingleAgentEnv:
    """Wraps SingleAgentAvoidanceEnv: every action passed to step() is
    filtered through the (single-threat) HOCBF check before physics
    executes it. Composition, not a fork -- fixes to
    SingleAgentAvoidanceEnv apply here automatically."""

    def __init__(self, env: SingleAgentAvoidanceEnv):
        self.env = env
        self.stats = FilterStats()
        self._last_position: np.ndarray | None = None
        self._last_velocity: np.ndarray | None = None

    def reset(self, initial_position_m: np.ndarray, initial_velocity_mps: np.ndarray) -> np.ndarray:
        self._last_position = np.array(initial_position_m, dtype=np.float64)
        self._last_velocity = np.array(initial_velocity_mps, dtype=np.float64)
        return self.env.reset(initial_position_m, initial_velocity_mps)

    def step(self, proposed_action: np.ndarray) -> tuple[np.ndarray, StepResult]:
        threat = Threat(
            relative_position_m=self._last_position,
            relative_velocity_mps=self._last_velocity,
            safety_radius_m=self.env.collision_radius_m,
        )
        filter_result = multi_threat_safe_action(
            threats=[threat],
            proposed_action_mps2=proposed_action,
            mean_motion=self.env.mean_motion,
            max_thrust_mps2=self.env.max_thrust_mps2,
        )

        self.stats.total_actions += 1
        if filter_result.was_corrected:
            self.stats.corrected_actions += 1
        if not filter_result.solver_success:
            self.stats.infeasible_actions += 1

        obs, result = self.env.step(filter_result.safe_action)
        self._last_position = obs[:3]
        self._last_velocity = obs[3:6]

        return obs, result


class FilteredMultiAgentEnv:
    """Wraps MultiAgentAvoidanceEnv: every agent's action is filtered
    against BOTH the shared origin threat AND every other live agent,
    via the multi-threat QP solver -- the real, intended use case for
    that solver, now that agent-to-agent collision is modeled."""

    def __init__(self, env: MultiAgentAvoidanceEnv):
        self.env = env
        self.stats = FilterStats()
        self._last_positions: list[np.ndarray] = []
        self._last_velocities: list[np.ndarray] = []

    def reset(self, initial_positions_m: list[np.ndarray], initial_velocities_mps: list[np.ndarray]) -> list[np.ndarray]:
        self._last_positions = [np.array(p, dtype=np.float64) for p in initial_positions_m]
        self._last_velocities = [np.array(v, dtype=np.float64) for v in initial_velocities_mps]
        return self.env.reset(initial_positions_m, initial_velocities_mps)

    def step(self, proposed_actions: list[np.ndarray]) -> tuple[list[np.ndarray], MultiAgentStepResult]:
        filtered_actions = []
        for i, action in enumerate(proposed_actions):
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

            filter_result = multi_threat_safe_action(
                threats=threats, proposed_action_mps2=action,
                mean_motion=self.env.mean_motion, max_thrust_mps2=self.env.max_thrust_mps2,
            )
            self.stats.total_actions += 1
            if filter_result.was_corrected:
                self.stats.corrected_actions += 1
            if not filter_result.solver_success:
                self.stats.infeasible_actions += 1
            filtered_actions.append(filter_result.safe_action)

        obs_list, result = self.env.step(filtered_actions)

        for i, obs in enumerate(obs_list):
            self._last_positions[i] = obs[:3]
            self._last_velocities[i] = obs[3:6]

        return obs_list, result