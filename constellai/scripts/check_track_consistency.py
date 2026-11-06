import argparse
import numpy as np
from constellai.evaluation.scaled_env import ScaledMultiAgentEnv
from constellai.evaluation.scenarios import STREAM_FC_VAL, make_rng, sample_scenario, warning_config
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv
from constellai.tracking.tracker import TrackedMultiAgentEnv


def make_env(cfg, sigma_p, accel_std, coop=False):
    raw = MultiAgentAvoidanceEnv(n_agents=cfg.n_agents, mean_motion=cfg.mean_motion, dt=cfg.dt,
                                 collision_radius_m=cfg.collision_radius_m,
                                 max_thrust_mps2=cfg.max_thrust_mps2, max_steps=cfg.max_steps)
    scaled = ScaledMultiAgentEnv(raw, cfg.n_agents, cfg.max_thrust_mps2, cfg.dt)
    return TrackedMultiAgentEnv(scaled, sigma_p=sigma_p, process_accel_std=accel_std, noise_seed=0, neighbour_thrust_known=coop)


def actions(kind, n, rng):
    if kind == "zero":
        return [np.zeros(3) for _ in range(n)]
    if kind == "random":
        return [rng.uniform(-1, 1, 3) for _ in range(n)]
    return [np.ones(3) for _ in range(n)]  # constant full burn on every axis


def nees(env, i, j=None):
    mean, cov, truth = env.pair_track(i, j)
    e = truth - mean
    return float(e @ np.linalg.solve(cov, e))


def run(cfg, sigma_p, accel_std, kind, episodes, coop=False):
    env = make_env(cfg, sigma_p, accel_std, coop)
    prim, nb = [], []
    for ep in range(episodes):
        pos, vel = sample_scenario(make_rng(STREAM_FC_VAL, 0, ep), cfg)
        env.reset(pos, vel, episode_index=ep)
        rng = np.random.default_rng([99, ep])
        ep_p, ep_n, done = [], [], False
        while not done:
            live = env.raw.live_mask()
            _, res = env.step(actions(kind, cfg.n_agents, rng))
            for i in range(cfg.n_agents):
                if not live[i]:
                    continue
                ep_p.append(nees(env, i))
                for j in range(cfg.n_agents):
                    if j != i and live[j]:
                        ep_n.append(nees(env, i, j))
            done = all(res.dones)
        if ep_p:
            prim.append(np.mean(ep_p))
        if ep_n:
            nb.append(np.mean(ep_n))
    return prim, nb


def fmt(v):
    v = np.asarray(v)
    return f"{v.mean():7.2f} +/- {2 * v.std(ddof=1) / np.sqrt(len(v)):5.2f}" if len(v) > 1 else "   n/a"


parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=30)
parser.add_argument("--accel-std", type=float, default=0.01 / np.sqrt(3.0))
parser.add_argument("--coop", action="store_true")
args = parser.parse_args()
cfg = warning_config("short")
print(f"accel_std={args.accel_std:.5f}, episodes={args.episodes}; target NEES = 6 (per-episode mean +/- 2 SE)")
print(f"{'sigma_p':>8} {'policy':>8} {'primary':>20} {'neighbour':>20}")
for sigma_p in (5.0, 25.0, 50.0):
    for kind in ("zero", "random", "burn"):
        p, n = run(cfg, sigma_p, args.accel_std, kind, args.episodes, args.coop)
        print(f"{sigma_p:8.1f} {kind:>8} {fmt(p):>20} {fmt(n):>20}", flush=True)
