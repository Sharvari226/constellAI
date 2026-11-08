"""Evaluate tracked-observation checkpoints (deterministic mean action) and reference policies.

Pilot stream by default; the final test stream (2) requires --final. Every policy gets a
fresh environment with the same noise seed, so noise sequences and scenarios are paired.
Raw per-trial results are saved beside the summary.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from constellai.evaluation.scaled_env import POLICY_ACTION_LIMIT, ScaledMultiAgentEnv
from constellai.evaluation.scenarios import STREAM_EVAL, warning_config
from constellai.evaluation.trials import ConstantBurnPolicy, ZeroPolicy, run_trials, summarize_trials
from constellai.forecasting.events import AnalyticForecaster
from constellai.models.marl.checkpoint import build_networks
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv
from constellai.tracking.tracker import TrackedMultiAgentEnv

PILOT_STREAM = 6


class MeanActionPolicy:
    def __init__(self, name, actor):
        self.name = name
        self.actor = actor

    def act(self, observations):
        with torch.no_grad():
            return [self.actor(torch.from_numpy(np.asarray(o)).float()).mean.numpy().astype(np.float64)
                    for o in observations]


def make_env(cfg, sigma_p, forecast, mission, noise_seed, mc_samples=500, fuel_weight=0.1):
    raw = MultiAgentAvoidanceEnv(n_agents=cfg.n_agents, mean_motion=cfg.mean_motion, dt=cfg.dt,
                                 collision_radius_m=cfg.collision_radius_m,
                                 max_thrust_mps2=cfg.max_thrust_mps2, max_steps=cfg.max_steps)
    raw.mission_mode = mission
    scaled = ScaledMultiAgentEnv(raw, cfg.n_agents, cfg.max_thrust_mps2, cfg.dt, fuel_weight=fuel_weight)
    fc = AnalyticForecaster(cfg.mean_motion, cfg.dt, cfg.collision_radius_m, n_samples=mc_samples) \
        if forecast == "analytic" else None
    oracle = AnalyticForecaster(cfg.mean_motion, cfg.dt, cfg.collision_radius_m, n_samples=1) if forecast == "oracle" else None
    return TrackedMultiAgentEnv(scaled, sigma_p=sigma_p, process_accel_std=0.0, noise_seed=noise_seed,
                                neighbour_thrust_known=True, forecaster=fc, oracle=oracle)


def load_actor(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    nets = build_networks(payload["rung"], payload["n_agents"], payload["max_thrust_mps2"],
                          payload.get("actor_obs_dim", 7), payload.get("critic_obs_dim", 7))
    for name, net in nets.items():
        net.load_state_dict(payload["state_dicts"][name])
        net.eval()
    return nets["actor"], payload["meta"], payload.get("actor_obs_dim", 7)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="*", default=[])
    p.add_argument("--baselines", action="store_true", help="also run zero-thrust and full-burn")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--preset", choices=["short", "long"], default="short")
    p.add_argument("--n-agents", type=int, default=3)
    p.add_argument("--sigma-p", type=float, default=None, help="default: the checkpoint's own value")
    p.add_argument("--noise-seed", type=int, default=12345)
    p.add_argument("--final", action="store_true", help="use the final test stream (2)")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    stream = STREAM_EVAL if args.final else PILOT_STREAM
    cfg = warning_config(args.preset, n_agents=args.n_agents)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for ck in args.ckpt:
        actor, meta, obs_dim = load_actor(ck)
        sigma = args.sigma_p if args.sigma_p is not None else meta["sigma_p"]
        jobs.append((Path(ck).parent.name, MeanActionPolicy(Path(ck).parent.name, actor),
                     dict(sigma_p=sigma, forecast=meta["forecast"], mission=meta["mission"]), obs_dim))
    if args.baselines:
        base = dict(sigma_p=args.sigma_p if args.sigma_p is not None else 25.0, forecast="none", mission="nominal")
        jobs.append(("zero_thrust", ZeroPolicy(), base, None))
        jobs.append(("constant_burn", ConstantBurnPolicy(POLICY_ACTION_LIMIT), base, None))

    print(f"stream {stream} ({'FINAL TEST' if args.final else 'pilot'}), preset {args.preset}, "
          f"{args.trials} trials, noise seed {args.noise_seed}")
    print(f"{'policy':<28}{'collision-free [95% CI]':>26}{'min sep m':>11}{'dv/agent':>10}{'saturation':>11}")
    for name, policy, env_kw, obs_dim in jobs:
        env = make_env(cfg, noise_seed=args.noise_seed, **env_kw)
        if obs_dim is not None:
            assert env.obs_dim == obs_dim, f"{name}: env obs {env.obs_dim} != checkpoint actor {obs_dim}"
        trials = run_trials(env, policy, cfg, args.trials, seed=0, stream=stream)
        s = summarize_trials(trials)
        print(f"{name:<28}{s['collision_free_rate']:7.3f} [{s['ci95_low']:5.3f},{s['ci95_high']:5.3f}]"
              f"{s['mean_min_separation_m']:11.1f}{s['mean_fuel_used_per_agent']:10.3f}{s['mean_saturation_fraction']:11.3f}",
              flush=True)
        (out_dir / f"eval_{name}_stream{stream}.json").write_text(json.dumps(
            {"policy": name, "stream": stream, "preset": args.preset, "noise_seed": args.noise_seed,
             "env": env_kw, "summary": s, "trials": [asdict(t) for t in trials]}, default=float))


if __name__ == "__main__":
    main()
