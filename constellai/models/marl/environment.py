"""M4's single-agent RL environment (baseline rung 1 of the ladder):
one chaser satellite, HCW relative dynamics, deciding whether/how to
apply a thrust maneuver to avoid a stationary(-in-LVLH) target while
conserving fuel.

Deliberately single-agent first -- this is the environment
Bourriez et al.'s POMDP baseline is echoing, and every later rung
(IPPO, MAPPO+penalty, MADDPG, constrained) needs this to exist and be
correct before it means anything to compare against.

Reward is returned as TWO separate values (mission_reward, safety_cost),
not combined into one scalar -- this is the concrete environment-level
commitment to the CMDP formulation the project settled on: safety must
remain a constraint an algorithm enforces, not a term folded into one
reward number an algorithm merely tries to maximize. Combining them
here would silently reintroduce the reward-penalty anti-pattern this
project explicitly moved away from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constellai.orbital_mechanics.relative_motion import (
    RelativeState,
    propagate_hcw_step,
)

# Placeholders, same status as every other threshold constant in this
# project: not yet through the Step-1 spec freeze. Do not treat as
# scientifically final.
DEFAULT_COLLISION_RADIUS_M = 50.0
DEFAULT_MAX_THRUST_MPS2 = 0.01
DEFAULT_FUEL_BUDGET = 100.0  # abstract units; consumed proportional to |thrust| * dt
DEFAULT_MAX_STEPS = 200


@dataclass
class StepResult:
    """One environment step's outcome.

    mission_reward : float
        Positive signal for task progress (here: staying near a target
        hold position -- a stand-in for "maintain station" until a real
        mission-utility definition is frozen).
    safety_cost : float
        NON-negative; 0 if outside the collision radius, 1.0 if a
        collision occurred this step. This is what a constrained
        algorithm's P(collision) <= epsilon check operates on -- it is
        NOT meant to be subtracted from mission_reward by this
        environment. That composition decision belongs to whichever
        algorithm (reward-penalty baseline vs. constrained method)
        consumes this environment, not to the environment itself.
    done : bool
    info : dict
    """

    mission_reward: float
    safety_cost: float
    done: bool
    info: dict


class SingleAgentAvoidanceEnv:
    """One chaser satellite under HCW dynamics, deciding thrust each step.

    State (returned by reset()/step()): np.ndarray, shape (7,)
        [rel_x, rel_y, rel_z, rel_vx, rel_vy, rel_vz, fuel_remaining]
        positions in meters, velocities in m/s, fuel in abstract units.

    Action (passed to step()): np.ndarray, shape (3,)
        Desired thrust acceleration [ax, ay, az], m/s^2 -- clipped to
        DEFAULT_MAX_THRUST_MPS2 in magnitude; the environment enforces
        this physical limit, it does not trust the caller to respect it.
    """

    def __init__(
        self,
        mean_motion: float,
        dt: float = 10.0,
        collision_radius_m: float = DEFAULT_COLLISION_RADIUS_M,
        max_thrust_mps2: float = DEFAULT_MAX_THRUST_MPS2,
        fuel_budget: float = DEFAULT_FUEL_BUDGET,
        max_steps: int = DEFAULT_MAX_STEPS,
    ):
        self.mean_motion = mean_motion
        self.dt = dt
        self.collision_radius_m = collision_radius_m
        self.max_thrust_mps2 = max_thrust_mps2
        self.fuel_budget = fuel_budget
        self.max_steps = max_steps

        self._state: RelativeState | None = None
        self._fuel: float = fuel_budget
        self._step_count: int = 0

    def reset(self, initial_position_m: np.ndarray, initial_velocity_mps: np.ndarray) -> np.ndarray:
        """Start a new episode from a caller-specified relative state.

        Parameters
        ----------
        initial_position_m : np.ndarray, shape (3,)
        initial_velocity_mps : np.ndarray, shape (3,)

        Returns
        -------
        np.ndarray, shape (7,) -- the initial observation.
        """
        self._state = RelativeState(
            position=np.array(initial_position_m, dtype=np.float64),
            velocity=np.array(initial_velocity_mps, dtype=np.float64),
        )
        self._fuel = self.fuel_budget
        self._step_count = 0
        return self._observation()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, StepResult]:
        """Apply one control action, advance dynamics by dt, return the
        new observation and a StepResult.

        Raises
        ------
        RuntimeError
            If called before reset().
        """
        if self._state is None:
            raise RuntimeError("step() called before reset()")

        thrust = np.clip(action, -self.max_thrust_mps2, self.max_thrust_mps2)
        fuel_cost = float(np.linalg.norm(thrust)) * self.dt
        self._fuel = max(0.0, self._fuel - fuel_cost)

        # Out of fuel: thrust physically cannot be applied, regardless
        # of what the policy requested -- this is a hard physical fact,
        # not a soft penalty for the algorithm to learn around.
        if self._fuel <= 0.0 and fuel_cost > 0:
            thrust = np.zeros(3)

        self._state = propagate_hcw_step(self._state, thrust, self.mean_motion, self.dt)
        self._step_count += 1

        separation_m = float(np.linalg.norm(self._state.position))
        collided = separation_m < self.collision_radius_m

        mission_reward = self._mission_reward(separation_m)
        safety_cost = 1.0 if collided else 0.0

        done = collided or self._fuel <= 0.0 or self._step_count >= self.max_steps

        info = {
            "separation_m": separation_m,
            "fuel_remaining": self._fuel,
            "collided": collided,
            "step_count": self._step_count,
        }

        return self._observation(), StepResult(mission_reward, safety_cost, done, info)

    def _mission_reward(self, separation_m: float) -> float:
        """Placeholder mission objective: reward staying near a nominal
        hold distance, penalize drifting far away. This is NOT the
        project's final mission-utility definition (that's Step-1 spec
        work) -- it exists so this environment is runnable and testable
        end-to-end today, not to pre-empt that design decision."""
        target_distance_m = 1000.0
        return -abs(separation_m - target_distance_m) / target_distance_m

    def _observation(self) -> np.ndarray:
        return np.concatenate([self._state.position, self._state.velocity, [self._fuel]])