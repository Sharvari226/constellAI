"""M5 integration: wraps M4's environments so every proposed action is
passed through the HOCBF safety filter before physics executes it.

Honest scope note: both environments currently model exactly ONE threat
per agent (the shared target/threat at the LVLH origin) -- there is no
agent-to-agent collision modeling yet. That means the single-threat
closed-form filter (cbf_filter.py) is what actually applies here; the
multi-threat QP solver (multi_threat_filter.py) has nothing to be
genuinely exercised against until agent-to-agent collision is added to
the environment itself, which is a separate, not-yet-done extension.
Wrapping with the multi-threat solver here, before that extension
exists, would be decorative wiring, not real integration -- so this
module deliberately uses the single-threat filter, honestly matching
the physics the environment actually models today.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constellai.models.marl.environment import SingleAgentAvoidanceEnv, StepResult
from constellai.models.marl.multi_agent_environment import (
    MultiAgentAvoidanceEnv,
    MultiAgentStepResult,
)
from constellai.safety.cbf_filter import hocbf_safe_action


@dataclass
class FilterStats:
    """Running correction-rate tracker -- this is the number the
    project commits to reporting (empirical correction rate), never
    "guaranteed safety" as a claim."""

    total_actions: int = 0
    corrected_actions: int = 0

    @property
    def correction_rate(self) -> float:
        return self.corrected_actions / self.total_actions if self.total_actions else 0.0


class FilteredSingleAgentEnv:
    """Wraps SingleAgentAvoidanceEnv: every action passed to step() is
    filtered through the single-threat HOCBF check before physics
    executes it. The underlying environment is unmodified -- this is
    composition, not a fork, so any fix to SingleAgentAvoidanceEnv
    automatically applies here too."""

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
        filter_result = hocbf_safe_action(
            relative_position_m=self._last_position,
            relative_velocity_mps=self._last_velocity,
            proposed_action_mps2=proposed_action,
            mean_motion=self.env.mean_motion,
            safety_radius_m=self.env.collision_radius_m,
        )

        self.stats.total_actions += 1
        if filter_result.was_corrected:
            self.stats.corrected_actions += 1

        obs, result = self.env.step(filter_result.safe_action)
        # Track state for the NEXT call's filter input -- obs is
        # [pos(3), vel(3), fuel(1)], matching environment.py's contract.
        self._last_position = obs[:3]
        self._last_velocity = obs[3:6]

        return obs, result


class FilteredMultiAgentEnv:
    """Same wrapping pattern as FilteredSingleAgentEnv, applied per
    agent -- each agent independently filtered against its own single
    threat (the shared origin), matching what MultiAgentAvoidanceEnv
    actually models. See module docstring for why this is the honest
    choice over the multi-threat solver at this stage."""

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
            filter_result = hocbf_safe_action(
                relative_position_m=self._last_positions[i],
                relative_velocity_mps=self._last_velocities[i],
                proposed_action_mps2=action,
                mean_motion=self.env.mean_motion,
                safety_radius_m=self.env.collision_radius_m,
            )
            self.stats.total_actions += 1
            if filter_result.was_corrected:
                self.stats.corrected_actions += 1
            filtered_actions.append(filter_result.safe_action)

        obs_list, result = self.env.step(filtered_actions)

        for i, obs in enumerate(obs_list):
            self._last_positions[i] = obs[:3]
            self._last_velocities[i] = obs[3:6]

        return obs_list, result