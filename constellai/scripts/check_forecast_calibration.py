"""Calibration of the analytic forecast fed by the filter, on the validation stream.

Per horizon: Brier vs the base-rate Brier, equal-mass-bin ECE, and the ECE expected for a
perfectly calibrated forecaster with the same predictions (a floor, since ECE is biased
upward). Confidence intervals are episode-clustered bootstraps.
"""

import argparse

import numpy as np

from constellai.evaluation.scaled_env import ScaledMultiAgentEnv
from constellai.evaluation.scenarios import STREAM_FC_VAL, make_rng, sample_scenario, warning_config
from constellai.forecasting.events import HORIZONS_S, AnalyticForecaster, cumulative_labels, pair_states
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv
from constellai.tracking.tracker import TrackedMultiAgentEnv


def ece_equal_mass(p, y, n_bins=10):
    bins = np.array_split(np.argsort(p, kind="stable"), n_bins)
    n = len(p)
    return float(sum(len(b) / n * abs(y[b].mean() - p[b].mean()) for b in bins if len(b)))


def collect(cfg, sigma_p, episodes, stride, n_samples, policy):
    raw = MultiAgentAvoidanceEnv(n_agents=cfg.n_agents, mean_motion=cfg.mean_motion, dt=cfg.dt,
                                 collision_radius_m=cfg.collision_radius_m,
                                 max_thrust_mps2=cfg.max_thrust_mps2, max_steps=cfg.max_steps)
    scaled = ScaledMultiAgentEnv(raw, cfg.n_agents, cfg.max_thrust_mps2, cfg.dt)
    env = TrackedMultiAgentEnv(scaled, sigma_p=sigma_p, process_accel_std=0.0, noise_seed=0,
                               neighbour_thrust_known=True)
    fc = AnalyticForecaster(cfg.mean_motion, cfg.dt, cfg.collision_radius_m, n_samples=n_samples)
    data = []
    for ep in range(episodes):
        pos, vel = sample_scenario(make_rng(STREAM_FC_VAL, 0, ep), cfg)
        env.reset(pos, vel, episode_index=ep)
        rng = np.random.default_rng([7, ep])
        P, Y, step, done = [], [], 0, False
        while not done:
            live = env.raw.live_mask()
            if step % stride == 0:
                for i, j, x0 in pair_states(env.raw.agent_state_vectors()):
                    if not live[i] or (j >= 0 and not live[j]):
                        continue
                    mean, cov, _ = env.pair_track(i, None if j < 0 else j)
                    P.append(fc(mean, cov))
                    Y.append(cumulative_labels(fc.engine.first_entry(x0), fc.steps)[0])
            acts = [np.zeros(3) if policy == "zero" else rng.uniform(-1, 1, 3) for _ in range(cfg.n_agents)]
            _, res = env.step(acts)
            step += 1
            done = all(res.dones)
        if P:
            data.append((np.array(P), np.array(Y)))
    return data


def metrics(P, Y, k, rng):
    p, y = P[:, k], Y[:, k]
    base = y.mean()
    brier = float(np.mean((p - y) ** 2))
    base_brier = float(base * (1 - base))
    skill = 1.0 - brier / base_brier if base_brier > 0 else float("nan")
    return brier, base_brier, skill, ece_equal_mass(p, y)


def report(data, boot, seed=0):
    rng = np.random.default_rng(seed)
    P = np.concatenate([d[0] for d in data])
    Y = np.concatenate([d[1] for d in data])
    print(f"{'H(s)':>6} {'n':>6} {'rate':>6} {'Brier':>7} {'baseB':>7} {'skill [95% CI]':>22} {'ECE [95% CI]':>22} {'ECE floor':>9}")
    ok = True
    for k, h in enumerate(HORIZONS_S):
        brier, base_brier, skill, ece = metrics(P, Y, k, rng)
        floor = float(np.mean([ece_equal_mass(P[:, k], (rng.random(len(P)) < P[:, k]).astype(float))
                               for _ in range(20)]))
        skills, eces = [], []
        for _ in range(boot):
            pick = rng.integers(0, len(data), len(data))
            Pb = np.concatenate([data[i][0] for i in pick])
            Yb = np.concatenate([data[i][1] for i in pick])
            _, _, s_b, e_b = metrics(Pb, Yb, k, rng)
            skills.append(s_b)
            eces.append(e_b)
        s_lo, s_hi = np.nanpercentile(skills, [2.5, 97.5])
        e_lo, e_hi = np.percentile(eces, [2.5, 97.5])
        ok = ok and skill > 0
        print(f"{h:6.0f} {len(P):6d} {Y[:, k].mean():6.3f} {brier:7.4f} {base_brier:7.4f} "
              f"{skill:7.3f} [{s_lo:5.2f},{s_hi:5.2f}] {ece:7.4f} [{e_lo:5.3f},{e_hi:5.3f}] {floor:9.4f}")
    print("  Brier beats base rate at every horizon:", "PASS" if ok else "FAIL")


parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--stride", type=int, default=5)
parser.add_argument("--n-samples", type=int, default=2000)
parser.add_argument("--boot", type=int, default=200)
parser.add_argument("--policy", choices=["zero", "random"], default="zero")
args = parser.parse_args()
cfg = warning_config("short")
print(f"analytic Monte Carlo forecast, filter consistent (thrust broadcast), policy={args.policy}, "
      f"episodes={args.episodes}, stride={args.stride}, samples={args.n_samples}")
for sigma_p in (5.0, 25.0, 50.0):
    print(f"\nsigma_p = {sigma_p:.0f} m ({sigma_p / cfg.collision_radius_m:.1f} r_c)")
    report(collect(cfg, sigma_p, args.episodes, args.stride, args.n_samples, args.policy), args.boot)
