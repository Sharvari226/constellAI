# Seeded initial-condition sampler shared by M4 training and M6 evaluation.
#
# Scenarios are built BACKWARDS from a chosen time of closest approach (TCA):
# pick the state the agent will have at TCA (a miss vector perpendicular to
# its approach velocity), then integrate HCW backwards in time with zero
# thrust. Under zero thrust the agent is therefore guaranteed to reach that
# miss distance at that step, so the fraction of "collision course" agents
# is controlled directly instead of hoped for. A uniform random draw would
# almost never produce a collision at these distances, which would make any
# policy (even one that does nothing) look safe.
#
# Training and evaluation draw from disjoint RNG streams, keyed by
# (stream, seed, index), so no evaluation scenario can leak into training.

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from constellai.common.constants import EARTH_RADIUS_WGS72_KM, GM_EARTH_WGS72_KM3_S2
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv
from constellai.orbital_mechanics.relative_motion import RelativeState, propagate_hcw_step

REFERENCE_ALTITUDE_KM = 500.0  # same shell as the M2/M3 synthetic scenarios
STREAM_TRAIN = 1
STREAM_EVAL = 2


def reference_mean_motion(altitude_km: float = REFERENCE_ALTITUDE_KM) -> float:
    # rad/s, circular reference orbit, WGS-72 constants (single source: constants.py)
    semi_major_axis_km = EARTH_RADIUS_WGS72_KM + altitude_km
    return math.sqrt(GM_EARTH_WGS72_KM3_S2 / semi_major_axis_km**3)


@dataclass(frozen=True)
class ScenarioConfig:
    n_agents: int = 3
    mean_motion: float = reference_mean_motion()
    dt: float = 10.0
    collision_radius_m: float = 50.0
    max_thrust_mps2: float = 0.01
    max_steps: int = 200
    # fraction of agents placed on a zero-thrust collision course; 0.15 with
    # N=3 gives ~45% of zero-thrust TRIALS (any agent collides) and ~18% of agents
    p_collision_course: float = 0.15
    # approach speed at TCA; kept <= ~1 m/s so one step (10 s) is <= ~10 m
    # and a 50 m collision sphere cannot be tunnelled through between steps
    approach_speed_mps: tuple = (0.3, 1.0)
    time_to_tca_s: tuple = (500.0, 1500.0)
    # collision course: miss distance <= this fraction of the collision radius
    collision_miss_fraction: float = 0.8
    # non-collision course: miss distance in multiples of the collision radius
    near_miss_radii: tuple = (1.5, 4.0)
    # reject scenarios whose agents start closer than this to each other / the origin
    min_start_separation_radii: float = 3.0
    max_resample_attempts: int = 100


def make_rng(stream: int, seed: int, index: int) -> np.random.Generator:
    return np.random.default_rng([stream, seed, index])


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _sample_agent_state(rng, cfg: ScenarioConfig, collision_course: bool):
    speed = rng.uniform(*cfg.approach_speed_mps)
    steps = max(1, int(round(rng.uniform(*cfg.time_to_tca_s) / cfg.dt)))
    heading = _unit_vector(rng)

    # miss vector perpendicular to the approach velocity, so TCA is the
    # closest-approach point (to first order; HCW curvature over this
    # window is small)
    side = _unit_vector(rng)
    side = side - np.dot(side, heading) * heading
    side = side / np.linalg.norm(side)

    if collision_course:
        miss_m = rng.uniform(0.0, cfg.collision_miss_fraction) * cfg.collision_radius_m
    else:
        miss_m = rng.uniform(*cfg.near_miss_radii) * cfg.collision_radius_m

    state = RelativeState(position=miss_m * side, velocity=speed * heading)
    for _ in range(steps):
        state = propagate_hcw_step(state, np.zeros(3), cfg.mean_motion, -cfg.dt)
    return np.array(state.position, dtype=np.float64), np.array(state.velocity, dtype=np.float64)


def _start_is_separated(positions, cfg: ScenarioConfig) -> bool:
    min_sep = cfg.min_start_separation_radii * cfg.collision_radius_m
    if any(np.linalg.norm(p) < min_sep for p in positions):
        return False
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if np.linalg.norm(positions[i] - positions[j]) < min_sep:
                return False
    return True


def sample_scenario(rng: np.random.Generator, cfg: ScenarioConfig):
    # returns (positions_m, velocities_mps), each a list of (3,) arrays,
    # the exact format MultiAgentAvoidanceEnv.reset() expects
    for _ in range(cfg.max_resample_attempts):
        positions, velocities = [], []
        for _ in range(cfg.n_agents):
            collision_course = bool(rng.random() < cfg.p_collision_course)
            p, v = _sample_agent_state(rng, cfg, collision_course)
            positions.append(p)
            velocities.append(v)
        if _start_is_separated(positions, cfg):
            return positions, velocities
    raise RuntimeError("could not sample a separated scenario; loosen ScenarioConfig")


def no_thrust_collision_rate(cfg: ScenarioConfig, n_episodes: int, seed: int = 0, stream: int = STREAM_EVAL) -> dict:
    # calibration check: how dangerous is this distribution if nothing acts?
    env = MultiAgentAvoidanceEnv(
        n_agents=cfg.n_agents, mean_motion=cfg.mean_motion, dt=cfg.dt,
        collision_radius_m=cfg.collision_radius_m, max_thrust_mps2=cfg.max_thrust_mps2,
        max_steps=cfg.max_steps,
    )
    agent_episodes = 0
    agent_collisions = 0
    agent_agent_collisions = 0
    any_collisions = 0

    for index in range(n_episodes):
        positions, velocities = sample_scenario(make_rng(stream, seed, index), cfg)
        env.reset(positions, velocities)
        collided = [False] * cfg.n_agents
        with_agent = [False] * cfg.n_agents
        done = False
        while not done:
            _, result = env.step([np.zeros(3) for _ in range(cfg.n_agents)])
            for i, info in enumerate(result.infos):
                if info.get("collided"):
                    collided[i] = True
                if info.get("collided_with_agent"):
                    with_agent[i] = True
            done = all(result.dones)
        agent_episodes += cfg.n_agents
        agent_collisions += sum(collided)
        agent_agent_collisions += sum(with_agent)
        any_collisions += int(any(collided))

    return {
        "per_agent_collision_rate": agent_collisions / agent_episodes,
        "any_agent_collision_rate": any_collisions / n_episodes,
        "agent_agent_collision_share": agent_agent_collisions / max(agent_collisions, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Report the zero-thrust collision rate of the scenario distribution")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--p-collision-course", type=float, default=0.15)
    args = parser.parse_args()

    cfg = ScenarioConfig(n_agents=args.n_agents, p_collision_course=args.p_collision_course)
    stats = no_thrust_collision_rate(cfg, args.episodes)
    print(f"zero-thrust, {args.episodes} episodes, N={cfg.n_agents}, p_collision_course={cfg.p_collision_course}")
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    main()