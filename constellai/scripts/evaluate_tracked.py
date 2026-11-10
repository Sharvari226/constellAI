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

from constellai.evaluation.heuristic import ThresholdHeuristicPolicy
from constellai.evaluation.triggers import DistanceTriggerPolicy, HcwMissTriggerPolicy, ProbabilityTriggerPolicy
from constellai.evaluation.scaled_env import POLICY_ACTION_LIMIT, ScaledMultiAgentEnv
from constellai.evaluation.scenarios import STREAM_EVAL, STREAM_FC_VAL, warning_config
from constellai.evaluation.trials import ConstantBurnPolicy, ZeroPolicy, run_trials, summarize_trials
from constellai.forecasting.events import AnalyticForecaster, FirstEntryEngine
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


def make_env(cfg, sigma_p, forecast, mission, noise_seed, mc_samples=500, fuel_weight=0.1, mc_gate=None):
    raw = MultiAgentAvoidanceEnv(n_agents=cfg.n_agents, mean_motion=cfg.mean_motion, dt=cfg.dt,
                                 collision_radius_m=cfg.collision_radius_m,
                                 max_thrust_mps2=cfg.max_thrust_mps2, max_steps=cfg.max_steps)
    raw.mission_mode = mission
    scaled = ScaledMultiAgentEnv(raw, cfg.n_agents, cfg.max_thrust_mps2, cfg.dt, fuel_weight=fuel_weight)
    fc = AnalyticForecaster(cfg.mean_motion, cfg.dt, cfg.collision_radius_m, n_samples=mc_samples, gate_sigmas=mc_gate) \
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
    p.add_argument("--heuristic-tau", type=float, nargs="*", default=[], help="threshold-heuristic trigger radii (x collision radius)")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--preset", choices=["short", "long"], default="short")
    p.add_argument("--n-agents", type=int, default=3)
    p.add_argument("--sigma-p", type=float, default=None, help="default: the checkpoint's own value")
    p.add_argument("--noise-seed", type=int, default=12345)
    p.add_argument("--final", action="store_true", help="use the final test stream (2)")
    p.add_argument("--val", action="store_true", help="use the validation stream (4), for choosing thresholds")
    p.add_argument("--distance-trigger", type=float, nargs="*", default=[], help="A0: current-distance triggers (x collision radius)")
    p.add_argument("--hcw-tau", type=float, nargs="*", default=[], help="E-HCW: point-miss triggers with exact HCW propagation (x collision radius)")
    p.add_argument("--prob-trigger", type=float, nargs="*", default=[], help="forecast-probability triggers p*")
    p.add_argument("--prob-horizon", type=int, default=3, help="horizon index watched by --prob-trigger (3 = 600 s)")
    p.add_argument("--forecast-kind", choices=["analytic", "oracle"], default="analytic")
    p.add_argument("--mc-gate", type=float, default=0.0, help="sigma gate for the analytic forecaster (0 = off)")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    assert not (args.final and args.val), "choose one stream"

    stream = STREAM_EVAL if args.final else (STREAM_FC_VAL if args.val else PILOT_STREAM)
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
    for tau in args.heuristic_tau:
        hb = dict(sigma_p=args.sigma_p if args.sigma_p is not None else 25.0, forecast="none", mission="nominal")
        jobs.append((f"heuristic_tau{tau:g}", ThresholdHeuristicPolicy(tau, cfg.collision_radius_m), hb, None))
    for d in args.distance_trigger:
        jobs.append((f"dist_D{d:g}", DistanceTriggerPolicy(d, cfg.collision_radius_m),
                     dict(sigma_p=args.sigma_p if args.sigma_p is not None else 25.0, forecast="none", mission="nominal"), None))
    for tau in args.hcw_tau:
        jobs.append((f"hcw_tau{tau:g}",
                     HcwMissTriggerPolicy(tau, FirstEntryEngine(cfg.mean_motion, cfg.dt, cfg.collision_radius_m, 60)),
                     dict(sigma_p=args.sigma_p if args.sigma_p is not None else 25.0, forecast="none", mission="nominal"), None))
    for pstar in args.prob_trigger:
        jobs.append((f"prob_{args.forecast_kind}_p{pstar:g}_h{args.prob_horizon}",
                     ProbabilityTriggerPolicy(pstar, args.prob_horizon, kind=args.forecast_kind),
                     dict(sigma_p=args.sigma_p if args.sigma_p is not None else 25.0, forecast=args.forecast_kind, mission="nominal"), None))

    print(f"stream {stream} ({'FINAL TEST' if args.final else ('validation' if args.val else 'pilot')}), preset {args.preset}, "
          f"{args.trials} trials, noise seed {args.noise_seed}")
    print(f"{'policy':<28}{'collision-free [95% CI]':>26}{'min sep m':>11}{'dv/agent':>10}{'saturation':>11}")
    for name, policy, env_kw, obs_dim in jobs:
        env = make_env(cfg, noise_seed=args.noise_seed, mc_gate=args.mc_gate or None, **env_kw)
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
