"""Constrained MAPPO on tracked observations.

Same rollout, update and logging code as the ladder; only the environment differs: actor
observations are filtered tracks plus the agent's own slot error, and the critic receives
the privileged true state (13 numbers per agent).
"""

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from constellai.evaluation.scaled_env import POLICY_ACTION_LIMIT, ScaledMultiAgentEnv
from constellai.evaluation.scenarios import STREAM_TRAIN, make_rng, sample_scenario, warning_config
from constellai.forecasting.events import AnalyticForecaster
from constellai.models.marl import constrained_mappo
from constellai.models.marl.checkpoint import build_networks, save_checkpoint
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv
from constellai.models.marl.objective import PIDLagrange
from constellai.tracking.tracker import TrackedMultiAgentEnv

_spec = importlib.util.spec_from_file_location("train_m4_ladder", Path(__file__).with_name("train_m4_ladder.py"))
ladder = importlib.util.module_from_spec(_spec)
sys.modules["train_m4_ladder"] = ladder  # dataclasses look the module up by name
_spec.loader.exec_module(ladder)

PRIV_DIM = 13


def make_env(cfg, args, seed):
    raw = MultiAgentAvoidanceEnv(n_agents=cfg.n_agents, mean_motion=cfg.mean_motion, dt=cfg.dt,
                                 collision_radius_m=cfg.collision_radius_m,
                                 max_thrust_mps2=cfg.max_thrust_mps2, max_steps=cfg.max_steps)
    raw.mission_mode = args.mission
    scaled = ScaledMultiAgentEnv(raw, cfg.n_agents, cfg.max_thrust_mps2, cfg.dt, fuel_weight=args.fuel_weight)
    forecaster = None
    oracle = AnalyticForecaster(cfg.mean_motion, cfg.dt, cfg.collision_radius_m, n_samples=1) if args.forecast == "oracle" else None
    if args.forecast == "analytic":
        forecaster = AnalyticForecaster(cfg.mean_motion, cfg.dt, cfg.collision_radius_m, n_samples=args.mc_samples)
    return TrackedMultiAgentEnv(scaled, sigma_p=args.sigma_p, process_accel_std=0.0, noise_seed=seed,
                                neighbour_thrust_known=True, forecaster=forecaster, oracle=oracle,
                                reward_scale=args.reward_scale)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--preset", choices=["short", "long"], default="short")
    p.add_argument("--n-agents", type=int, default=3)
    p.add_argument("--sigma-p", type=float, default=25.0)
    p.add_argument("--mission", choices=["nominal", "primary_shell"], default="nominal")
    p.add_argument("--forecast", choices=["none", "analytic", "oracle"], default="none")
    p.add_argument("--mc-samples", type=int, default=500)
    p.add_argument("--reward-scale", type=float, default=1.0, help="fixed multiplier on mission rewards (critic-scale control)")
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--action-std", type=float, default=ladder.DEFAULT_ACTION_STD)
    p.add_argument("--fuel-weight", type=float, default=ladder.DEFAULT_FUEL_WEIGHT)
    p.add_argument("--initial-lambda", type=float, default=1.0)
    p.add_argument("--budget", type=float, default=None, help="per-agent discounted shaped-cost budget d; enables the PID multiplier")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = warning_config(args.preset, n_agents=args.n_agents)
    env = make_env(cfg, args, args.seed)
    pos, vel = sample_scenario(make_rng(STREAM_TRAIN, args.seed, 0), cfg)
    env.reset(pos, vel)
    assert env.privileged_state().shape == (cfg.n_agents * PRIV_DIM,), env.privileged_state().shape
    env._episode = 0  # the shape check must not shift the noise sequence

    networks = build_networks("constrained_mappo", cfg.n_agents, POLICY_ACTION_LIMIT,
                              actor_obs_dim=env.obs_dim, critic_obs_dim=PRIV_DIM)
    ladder._set_action_std(networks, args.action_std)
    actor, rc, cc = networks["actor"], networks["reward_critic"], networks["cost_critic"]
    a_opt, r_opt, c_opt = (torch.optim.Adam(m.parameters(), lr=args.lr) for m in (actor, rc, cc))
    log = ladder._make_logger("constrained_mappo", args.seed, args.log_every)

    lam, history = args.initial_lambda, []
    pid = PIDLagrange(budget=args.budget) if args.budget is not None else None
    print(f"tracked constrained_mappo: seed {args.seed}, {args.episodes} episodes, preset {args.preset}, "
          f"mission {args.mission}, sigma_p {args.sigma_p}, forecast {args.forecast}, actor obs {env.obs_dim}")
    for ep in range(args.episodes):
        positions, velocities = sample_scenario(make_rng(STREAM_TRAIN, args.seed, ep), cfg)
        trajs = constrained_mappo.collect_rollout(env, actor, rc, cc, positions, velocities)
        out = constrained_mappo.constrained_update(
            trajs, actor, rc, cc, lagrange_lambda=lam,
            actor_optimizer=a_opt, reward_critic_optimizer=r_opt, cost_critic_optimizer=c_opt, pid=pid,
        )
        lam = out["lambda"]
        row = ladder._ppo_row(ep, trajs, max(len(t.mission_rewards) for t in trajs))
        row.update(out)
        history.append(row)
        log(history)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"constrained_mappo_seed{args.seed}_history.json").write_text(json.dumps(history, default=float))
    meta = {"seed": args.seed, "episodes": args.episodes, "preset": args.preset, "mission": args.mission,
            "sigma_p": args.sigma_p, "forecast": args.forecast, "lr": args.lr, "action_std": args.action_std,
            "tracked": True}
    save_checkpoint(out_dir / f"constrained_mappo_seed{args.seed}.pt", "constrained_mappo", cfg.n_agents,
                    POLICY_ACTION_LIMIT, networks, meta, actor_obs_dim=env.obs_dim, critic_obs_dim=PRIV_DIM)

    k = min(10, len(history))
    med = lambda xs: float(np.median(xs))
    ev = [h["reward_explained_variance"] for h in history]
    cev = [h["cost_explained_variance"] for h in history]
    loss = [h["reward_value_loss"] for h in history]
    print(f"SUMMARY mission={args.mission} sigma_p={args.sigma_p}: reward EV first{k} {med(ev[:k]):.2f} "
          f"last{k} {med(ev[-k:]):.2f} | cost EV first{k} {med(cev[:k]):.2f} last{k} {med(cev[-k:]):.2f} | "
          f"reward loss last{k} median {med(loss[-k:]):.1f} | "
          f"collision episodes {sum(bool(h['any_collision']) for h in history)} | lambda end {history[-1]['lambda']:.3f} | observed cost last{k} median {med([h['observed_discounted_cost'] for h in history][-k:]):.2f}")


if __name__ == "__main__":
    main()
