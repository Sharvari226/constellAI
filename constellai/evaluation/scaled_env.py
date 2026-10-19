# Scaling wrappers for M4 training and evaluation (composition, no library edits).
#
# Phase 2 diagnostics showed why the unscaled setup could not learn:
#   - observations carried raw meters (hundreds to thousands) into tanh networks,
#   - policy outputs were O(0.1-1) against a 0.01 m/s^2 thrust limit, so every
#     action was clipped to full thrust (saturation ~1.0 for every untrained and
#     trained policy), and
#   - fuel cost nothing, so heavy burning was never penalised.
#
# This wrapper fixes all three at the interface:
#   - observations: positions / 1000 m, fuel / budget (velocities are already ~1 m/s),
#   - actions: the policy outputs a FRACTION of the thrust limit per axis, clipped
#     to [-1, 1] and scaled to m/s^2 before the real environment sees it,
#   - reward: a fuel term is subtracted from mission_reward, equal to fuel_weight
#     at a full three-axis burn. Collisions stay in safety_cost, untouched, so the
#     CMDP structure that rung 5 relies on is preserved.
#
# The fuel term is computed from the fuel the environment actually spent, so it
# stays correct when an M5 safety filter sits INSIDE this wrapper and changes
# the executed thrust. Infos pass through in physical units (meters, abstract fuel).

from __future__ import annotations

import math

import numpy as np

from constellai.models.marl.environment import DEFAULT_FUEL_BUDGET, StepResult
from constellai.models.marl.multi_agent_environment import MultiAgentStepResult

POSITION_SCALE_M = 1000.0
POLICY_ACTION_LIMIT = 1.0  # policies act in fractions of the thrust limit
DEFAULT_FUEL_WEIGHT = 0.1  # per step at a full three-axis burn; a 200-step full burn costs 20 vs 50 per collision step


def scale_observation(obs, fuel_budget: float) -> np.ndarray:
    scaled = np.array(obs, dtype=np.float64)
    scaled[:3] /= POSITION_SCALE_M
    scaled[6] /= fuel_budget
    return scaled


def _to_physical(action, max_thrust_mps2: float) -> np.ndarray:
    return np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0) * max_thrust_mps2


def _fuel_penalty(fuel_spent: float, max_thrust_mps2: float, dt: float, fuel_weight: float) -> float:
    # a full three-axis burn spends sqrt(3) * limit * dt per step -> penalty == fuel_weight
    return fuel_weight * fuel_spent / (math.sqrt(3.0) * max_thrust_mps2 * dt)


class ScaledMultiAgentEnv:
    action_limit = POLICY_ACTION_LIMIT

    def __init__(self, env, n_agents: int, max_thrust_mps2: float, dt: float,
                 fuel_budget: float = DEFAULT_FUEL_BUDGET, fuel_weight: float = DEFAULT_FUEL_WEIGHT):
        self.env = env
        self.n_agents = n_agents
        self.max_thrust_mps2 = max_thrust_mps2
        self.dt = dt
        self.fuel_budget = fuel_budget
        self.fuel_weight = fuel_weight
        self._last_fuel = [fuel_budget] * n_agents

    def reset(self, initial_positions_m, initial_velocities_mps):
        observations = self.env.reset(initial_positions_m, initial_velocities_mps)
        self._last_fuel = [self.fuel_budget] * self.n_agents
        return [scale_observation(o, self.fuel_budget) for o in observations]

    def step(self, actions):
        physical = [_to_physical(a, self.max_thrust_mps2) for a in actions]
        observations, result = self.env.step(physical)

        rewards = list(result.mission_rewards)
        for i, info in enumerate(result.infos):
            if info.get("already_done"):
                continue
            fuel_now = info["fuel_remaining"]
            spent = self._last_fuel[i] - fuel_now
            self._last_fuel[i] = fuel_now
            rewards[i] -= _fuel_penalty(spent, self.max_thrust_mps2, self.dt, self.fuel_weight)

        scaled = [scale_observation(o, self.fuel_budget) for o in observations]
        return scaled, MultiAgentStepResult(rewards, result.safety_costs, result.dones, result.infos)


class ScaledSingleAgentEnv:
    action_limit = POLICY_ACTION_LIMIT

    def __init__(self, env, max_thrust_mps2: float, dt: float,
                 fuel_budget: float = DEFAULT_FUEL_BUDGET, fuel_weight: float = DEFAULT_FUEL_WEIGHT):
        self.env = env
        self.max_thrust_mps2 = max_thrust_mps2
        self.dt = dt
        self.fuel_budget = fuel_budget
        self.fuel_weight = fuel_weight
        self._last_fuel = fuel_budget

    def reset(self, initial_position_m, initial_velocity_mps):
        observation = self.env.reset(initial_position_m, initial_velocity_mps)
        self._last_fuel = self.fuel_budget
        return scale_observation(observation, self.fuel_budget)

    def step(self, action):
        observation, result = self.env.step(_to_physical(action, self.max_thrust_mps2))
        fuel_now = result.info["fuel_remaining"]
        spent = self._last_fuel - fuel_now
        self._last_fuel = fuel_now
        reward = result.mission_reward - _fuel_penalty(spent, self.max_thrust_mps2, self.dt, self.fuel_weight)
        return scale_observation(observation, self.fuel_budget), StepResult(
            reward, result.safety_cost, result.done, result.info,
        )