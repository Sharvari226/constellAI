"""M4 rung 2's environment: N independent chasers, each running the
SAME SingleAgentAvoidanceEnv dynamics against a SHARED target/threat
position. This is deliberately NOT a new physics model -- it wraps N
copies of already-validated single-agent dynamics, so nothing here
needs re-validating against HCW periodicity; only the coordination
layer on top is new.

"Independent" here means literally that: each agent's observation is
its OWN relative state only -- no agent sees any other agent's state.
That is precisely what distinguishes IPPO (this rung) from MAPPO
(rung 3, which adds a centralized critic with access to joint state
during training). Building this environment to genuinely withhold
cross-agent information is what makes the IPPO-vs-MAPPO comparison
later mean something -- if this environment quietly leaked joint state
into each agent's observation, the two rungs would not actually be
testing different things.

Collision checking IS inherently joint (an agent can collide with the
shared target OR with another agent), so the environment computes that
centrally -- but only exposes each agent its own outcome, not the full
joint collision matrix, keeping the "independent" contract honest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constellai.models.marl.environment import (
    DEFAULT_COLLISION_RADIUS_M,
    DEFAULT_FUEL_BUDGET,
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_THRUST_MPS2,
)
from constellai.orbital_mechanics.relative_motion import RelativeState, propagate_hcw_step


@dataclass
class MultiAgentStepResult:
    """Per-agent outcome for one environment step.

    mission_rewards, safety_costs, dones : list[float] / list[bool]
        One entry per agent, same ordering as the agent list passed to
        reset(). Kept as separate per-agent lists, not a single
        combined array, so an IPPO implementation naturally only reads
        its own agent's slice -- accidentally reading another agent's
        entry is possible but requires deliberately indexing into it,
        not something that happens by default.
    """

    mission_rewards: list[float]
    safety_costs: list[float]
    dones: list[bool]
    infos: list[dict]


class MultiAgentAvoidanceEnv:
    """N independent chasers under HCW dynamics, each relative to a
    shared target/threat at the origin of the LVLH frame.

    Each agent's observation: np.ndarray, shape (7,) -- IDENTICAL format
    to SingleAgentAvoidanceEnv (rel pos/vel + own fuel). No agent's
    observation includes any other agent's state -- see module
    docstring for why that matters.
    """

    def __init__(
        self,
        n_agents: int,
        mean_motion: float,
        dt: float = 10.0,
        collision_radius_m: float = DEFAULT_COLLISION_RADIUS_M,
        max_thrust_mps2: float = DEFAULT_MAX_THRUST_MPS2,
        fuel_budget: float = DEFAULT_FUEL_BUDGET,
        max_steps: int = DEFAULT_MAX_STEPS,
    ):
        self.n_agents = n_agents
        self.mean_motion = mean_motion
        self.dt = dt
        self.collision_radius_m = collision_radius_m
        self.max_thrust_mps2 = max_thrust_mps2
        self.fuel_budget = fuel_budget
        self.max_steps = max_steps

        self._states: list[RelativeState] = []
        self._fuel: list[float] = []
        self._done: list[bool] = []
        self._step_count = 0

    def reset(self, initial_positions_m: list[np.ndarray], initial_velocities_mps: list[np.ndarray]) -> list[np.ndarray]:
        """Parameters mirror SingleAgentAvoidanceEnv.reset(), one entry
        per agent. Returns a list of (7,) observations, one per agent."""
        if len(initial_positions_m) != self.n_agents or len(initial_velocities_mps) != self.n_agents:
            raise ValueError("must provide exactly one initial position/velocity per agent")

        self._states = [
            RelativeState(position=np.array(p, dtype=np.float64), velocity=np.array(v, dtype=np.float64))
            for p, v in zip(initial_positions_m, initial_velocities_mps)
        ]
        self._fuel = [self.fuel_budget] * self.n_agents
        self._done = [False] * self.n_agents
        self._step_count = 0
        return self._observations()

    def step(self, actions: list[np.ndarray]) -> tuple[list[np.ndarray], MultiAgentStepResult]:
        """actions: list of (3,) thrust vectors, one per agent, in the
        SAME order as reset(). An agent already marked done from a
        previous step is not re-stepped -- its thrust is ignored and it
        keeps reporting done=True, mirroring standard multi-agent env
        convention (agents can finish at different times)."""
        if len(actions) != self.n_agents:
            raise ValueError(f"expected {self.n_agents} actions, got {len(actions)}")

        mission_rewards, safety_costs, infos = [], [], []
        self._step_count += 1

        for i in range(self.n_agents):
            if self._done[i]:
                mission_rewards.append(0.0)
                safety_costs.append(0.0)
                infos.append({"already_done": True})
                continue

            thrust = np.clip(actions[i], -self.max_thrust_mps2, self.max_thrust_mps2)
            fuel_cost = float(np.linalg.norm(thrust)) * self.dt
            self._fuel[i] = max(0.0, self._fuel[i] - fuel_cost)
            if self._fuel[i] <= 0.0 and fuel_cost > 0:
                thrust = np.zeros(3)

            self._states[i] = propagate_hcw_step(self._states[i], thrust, self.mean_motion, self.dt)

            separation_m = float(np.linalg.norm(self._states[i].position))
            collided = separation_m < self.collision_radius_m

            target_distance_m = 1000.0
            mission_reward = -abs(separation_m - target_distance_m) / target_distance_m
            safety_cost = 1.0 if collided else 0.0

            self._done[i] = collided or self._fuel[i] <= 0.0 or self._step_count >= self.max_steps

            mission_rewards.append(mission_reward)
            safety_costs.append(safety_cost)
            infos.append({
                "separation_m": separation_m,
                "fuel_remaining": self._fuel[i],
                "collided": collided,
                "step_count": self._step_count,
            })

        return self._observations(), MultiAgentStepResult(mission_rewards, safety_costs, list(self._done), infos)

    def _observations(self) -> list[np.ndarray]:
        return [
            np.concatenate([s.position, s.velocity, [f]])
            for s, f in zip(self._states, self._fuel)
        ]