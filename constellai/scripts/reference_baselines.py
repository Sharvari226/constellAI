"""Reference policies on the fixed env: baseline difficulty and the shaped-cost scale
(used to set the budget d). Pilot stream only; the final test stream (2) is untouched."""

import argparse

import numpy as np

from constellai.evaluation.scaled_env import ScaledMultiAgentEnv
from constellai.evaluation.scenarios import make_rng, sample_scenario, warning_config
from constellai.evaluation.stats import wilson_interval
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv

GAMMA = 0.99
PILOT_STREAM = 6


def run_episode(env, cfg, kind, rng, positions, velocities):
    env.reset(positions, velocities)
    n = cfg.n_agents
    disc, dv, collided = np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool)
    t, done = 0, False
    while not done:
        if kind == "zero":
            acts = [np.zeros(3) for _ in range(n)]
        elif kind == "burn":
            acts = [np.ones(3) for _ in range(n)]
        else:
            acts = [rng.uniform(-1.0, 1.0, 3) for _ in range(n)]
        _, res = env.step(acts)
        for i, info in enumerate(res.infos):
            if info.get("already_done"):
                continue
            disc[i] += (GAMMA ** t) * res.safety_costs[i]
            dv[i] += float(np.linalg.norm(info["thrust_mps2"])) * cfg.dt
            collided[i] = collided[i] or bool(info["collided"])
        t += 1
        done = all(res.dones)
    return disc, dv, collided


parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=100)
args = parser.parse_args()

for preset in ("short", "long"):
    cfg = warning_config(preset)
    raw = MultiAgentAvoidanceEnv(n_agents=cfg.n_agents, mean_motion=cfg.mean_motion, dt=cfg.dt,
                                 collision_radius_m=cfg.collision_radius_m,
                                 max_thrust_mps2=cfg.max_thrust_mps2, max_steps=cfg.max_steps)
    env = ScaledMultiAgentEnv(raw, cfg.n_agents, cfg.max_thrust_mps2, cfg.dt)
    print(f"\n{preset} warning {cfg.time_to_tca_s}, {args.episodes} episodes, stream {PILOT_STREAM}, gamma={GAMMA}")
    print(f"{'policy':>7} {'collision-free [95% CI]':>26} {'agent coll.':>11} {'dv/agent m/s':>13} {'disc cost mean':>15} {'p90':>7}")
    for kind in ("zero", "random", "burn"):
        discs, dvs, cols, any_col = [], [], [], 0
        for ep in range(args.episodes):
            pos, vel = sample_scenario(make_rng(PILOT_STREAM, 0, ep), cfg)
            d, v, c = run_episode(env, cfg, kind, np.random.default_rng([11, ep]), pos, vel)
            discs.extend(d)
            dvs.extend(v)
            cols.extend(c)
            any_col += int(c.any())
        free = args.episodes - any_col
        lo, hi = wilson_interval(free, args.episodes)
        print(f"{kind:>7} {free / args.episodes:7.3f} [{lo:5.3f},{hi:5.3f}] {np.mean(cols):11.3f} "
              f"{np.mean(dvs):13.3f} {np.mean(discs):15.4f} {np.percentile(discs, 90):7.3f}", flush=True)
