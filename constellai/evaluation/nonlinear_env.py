# Independent ground-truth simulator for M6 (Phase 3).
#
# Why this exists: M4 policies train on, and M5 reasons with, the linearised
# Hill-Clohessy-Wiltshire (HCW) model. Evaluating them inside that same model is
# circular. This simulator integrates the NONLINEAR two-body problem in an
# inertial frame and shares no dynamics code with orbital_mechanics/relative_motion.py:
#   - the chief (the shared target at the LVLH origin) flies an exact circular orbit,
#   - every agent is a full inertial two-body state, integrated with RK4,
#   - thrust is commanded in the LVLH frame (as in the HCW environment) and rotated
#     into the inertial frame at every integration stage,
#   - states are converted back to LVLH (radial x, along-track y, cross-track z)
#     only to build observations and to measure separations.
#
# Scope: point-mass gravity of a spherical Earth (no J2, drag or solar pressure).
# That is the same physical model HCW linearises, so any disagreement with the HCW
# environment is genuine nonlinearity (second order in separation / orbital radius),
# not unmodelled physics. Differences grow as separation squared, which the tests
# check.
#
# The class mirrors MultiAgentAvoidanceEnv's public interface and attributes
# (reset, step, get_threat_positions_and_velocities, collision_radius_m,
# max_thrust_mps2, mean_motion, n_agents, ...), so the trial runner, the scaling
# wrapper and the M5 FilteredMultiAgentEnv all work on it unchanged.

from __future__ import annotations

import numpy as np

from constellai.common.constants import GM_EARTH_WGS72_KM3_S2
from constellai.models.marl.environment import (
    DEFAULT_COLLISION_RADIUS_M,
    DEFAULT_FUEL_BUDGET,
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_THRUST_MPS2,
)
from constellai.models.marl.multi_agent_environment import MultiAgentStepResult

MU_M3_S2 = GM_EARTH_WGS72_KM3_S2 * 1e9


def reference_radius_m(mean_motion: float) -> float:
    # radius of the circular reference orbit with this mean motion
    return (MU_M3_S2 / mean_motion**2) ** (1.0 / 3.0)


def lvlh_rotation(theta: float) -> np.ndarray:
    # columns are the LVLH axes in the inertial frame: radial, along-track, orbit normal
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def chief_state(radius_m: float, mean_motion: float, t: float):
    theta = mean_motion * t
    position = radius_m * np.array([np.cos(theta), np.sin(theta), 0.0])
    velocity = radius_m * mean_motion * np.array([-np.sin(theta), np.cos(theta), 0.0])
    return position, velocity


def lvlh_to_eci(rho, rho_dot, radius_m: float, mean_motion: float, t: float):
    # rho_dot is the velocity seen in the rotating frame, so add the frame's rotation
    rotation = lvlh_rotation(mean_motion * t)
    chief_r, chief_v = chief_state(radius_m, mean_motion, t)
    omega_cross_rho = mean_motion * np.array([-rho[1], rho[0], 0.0])
    return chief_r + rotation @ rho, chief_v + rotation @ (rho_dot + omega_cross_rho)


def eci_to_lvlh(r, v, radius_m: float, mean_motion: float, t: float):
    rotation = lvlh_rotation(mean_motion * t)
    chief_r, chief_v = chief_state(radius_m, mean_motion, t)
    rho = rotation.T @ (r - chief_r)
    omega_cross_rho = mean_motion * np.array([-rho[1], rho[0], 0.0])
    rho_dot = rotation.T @ (v - chief_v) - omega_cross_rho
    return rho, rho_dot


def _derivative(state, t, thrust_lvlh, mean_motion):
    r, v = state[:3], state[3:]
    gravity = -MU_M3_S2 * r / np.linalg.norm(r) ** 3
    thrust = lvlh_rotation(mean_motion * t) @ thrust_lvlh
    return np.concatenate([v, gravity + thrust])


def propagate_eci_step(state, t, dt, thrust_lvlh, mean_motion, substeps=2):
    # RK4 on the inertial two-body state; thrust is held fixed in the LVLH frame
    h = dt / substeps
    for k in range(substeps):
        tk = t + k * h
        k1 = _derivative(state, tk, thrust_lvlh, mean_motion)
        k2 = _derivative(state + h / 2 * k1, tk + h / 2, thrust_lvlh, mean_motion)
        k3 = _derivative(state + h / 2 * k2, tk + h / 2, thrust_lvlh, mean_motion)
        k4 = _derivative(state + h * k3, tk + h, thrust_lvlh, mean_motion)
        state = state + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return state


class NonlinearAvoidanceEnv:
    def __init__(
        self,
        n_agents: int,
        mean_motion: float,
        dt: float = 10.0,
        collision_radius_m: float = DEFAULT_COLLISION_RADIUS_M,
        max_thrust_mps2: float = DEFAULT_MAX_THRUST_MPS2,
        fuel_budget: float = DEFAULT_FUEL_BUDGET,
        max_steps: int = DEFAULT_MAX_STEPS,
        substeps: int = 2,
    ):
        self.n_agents = n_agents
        self.mean_motion = mean_motion
        self.dt = dt
        self.collision_radius_m = collision_radius_m
        self.max_thrust_mps2 = max_thrust_mps2
        self.fuel_budget = fuel_budget
        self.max_steps = max_steps
        self.substeps = substeps
        self.radius_m = reference_radius_m(mean_motion)

        self._eci: list = []  # inertial [r, v] per agent
        self._rho: list = []  # last LVLH position per agent
        self._rho_dot: list = []
        self._fuel: list = []
        self._done: list = []
        self._step_count = 0

    def reset(self, initial_positions_m, initial_velocities_mps):
        if len(initial_positions_m) != self.n_agents or len(initial_velocities_mps) != self.n_agents:
            raise ValueError("must provide exactly one initial position/velocity per agent")
        self._rho = [np.array(p, dtype=np.float64) for p in initial_positions_m]
        self._rho_dot = [np.array(v, dtype=np.float64) for v in initial_velocities_mps]
        self._eci = []
        for rho, rho_dot in zip(self._rho, self._rho_dot):
            r, v = lvlh_to_eci(rho, rho_dot, self.radius_m, self.mean_motion, 0.0)
            self._eci.append(np.concatenate([r, v]))
        self._fuel = [self.fuel_budget] * self.n_agents
        self._done = [False] * self.n_agents
        self._step_count = 0
        return self._observations()

    def get_threat_positions_and_velocities(self, agent_id: int):
        # other live agents as seen from agent_id, in LVLH (same convention as the HCW environment)
        threats = []
        for j in range(self.n_agents):
            if j == agent_id or self._done[j]:
                continue
            threats.append((self._rho[j] - self._rho[agent_id], self._rho_dot[j] - self._rho_dot[agent_id]))
        return threats

    def step(self, actions):
        if len(actions) != self.n_agents:
            raise ValueError(f"expected {self.n_agents} actions, got {len(actions)}")

        t_start = self._step_count * self.dt
        t_end = t_start + self.dt
        self._step_count += 1

        # advance every live agent first, so collision checks use one consistent time
        for i in range(self.n_agents):
            if self._done[i]:
                continue
            thrust = np.clip(np.asarray(actions[i], dtype=np.float64), -self.max_thrust_mps2, self.max_thrust_mps2)
            fuel_cost = float(np.linalg.norm(thrust)) * self.dt
            self._fuel[i] = max(0.0, self._fuel[i] - fuel_cost)
            if self._fuel[i] <= 0.0 and fuel_cost > 0:
                thrust = np.zeros(3)
            self._eci[i] = propagate_eci_step(
                self._eci[i], t_start, self.dt, thrust, self.mean_motion, self.substeps,
            )
            self._rho[i], self._rho_dot[i] = eci_to_lvlh(
                self._eci[i][:3], self._eci[i][3:], self.radius_m, self.mean_motion, t_end,
            )

        mission_rewards, safety_costs, infos = [], [], []
        newly_done = list(self._done)
        for i in range(self.n_agents):
            if self._done[i]:
                mission_rewards.append(0.0)
                safety_costs.append(0.0)
                infos.append({"already_done": True})
                continue

            separation_to_origin = float(np.linalg.norm(self._rho[i]))
            collided_with_origin = separation_to_origin < self.collision_radius_m

            collided_with_agent = False
            nearest_agent = float("inf")
            for j in range(self.n_agents):
                if j == i or self._done[j]:
                    continue
                separation = float(np.linalg.norm(self._rho[i] - self._rho[j]))
                nearest_agent = min(nearest_agent, separation)
                if separation < self.collision_radius_m:
                    collided_with_agent = True

            collided = collided_with_origin or collided_with_agent
            separation_m = min(separation_to_origin, nearest_agent)
            mission_rewards.append(-abs(separation_m - 1000.0) / 1000.0)
            safety_costs.append(1.0 if collided else 0.0)
            newly_done[i] = collided or self._fuel[i] <= 0.0 or self._step_count >= self.max_steps
            infos.append({
                "separation_m": separation_m,
                "separation_to_origin_m": separation_to_origin,
                "collided_with_origin": collided_with_origin,
                "collided_with_agent": collided_with_agent,
                "fuel_remaining": self._fuel[i],
                "collided": collided,
                "step_count": self._step_count,
            })

        self._done = newly_done
        return self._observations(), MultiAgentStepResult(mission_rewards, safety_costs, list(self._done), infos)

    def _observations(self):
        return [
            np.concatenate([rho, rho_dot, [fuel]])
            for rho, rho_dot, fuel in zip(self._rho, self._rho_dot, self._fuel)
        ]