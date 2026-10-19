# M6: train the five M4 rungs on one shared scenario distribution and save an
# evaluation-ready checkpoint per (rung, seed).
#
# Run: python -m constellai.scripts.train_m4_ladder --rung ippo mappo --seeds 0 --episodes 500
#      python -m constellai.scripts.train_m4_ladder --rung all --seeds 0 1 2 --episodes 500
#
# Notes that matter when reading the results:
#   - Every environment is wrapped in the scaling wrapper (evaluation/scaled_env.py):
#     observations are rescaled, policies act in FRACTIONS of the thrust limit, and a
#     fuel term (--fuel-weight) is subtracted from mission_reward. Collisions stay in
#     safety_cost. Checkpoints trained without the wrapper (outputs/m4) are not
#     compatible with this setup.
#   - Rung 1 (REINFORCE) trains single-agent; later phases deploy that policy
#     independently on each agent. Rungs 2-5 train on N agents together.
#   - Collision rates logged here are under the stochastic / exploration policy,
#     so they describe learning progress, not final safety. Final numbers come from
#     evaluation on the held-out EVAL scenario stream.
#   - Exploration scale is matched across rungs (--action-std, in fractions of the
#     thrust limit, default 0.3).

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from constellai.evaluation.scaled_env import (
    DEFAULT_FUEL_WEIGHT,
    POLICY_ACTION_LIMIT,
    POSITION_SCALE_M,
    ScaledMultiAgentEnv,
    ScaledSingleAgentEnv,
)
from constellai.evaluation.scenarios import STREAM_TRAIN, ScenarioConfig, make_rng, sample_scenario
from constellai.models.marl import constrained_mappo, ippo, maddpg, mappo, single_agent_baseline
from constellai.models.marl.checkpoint import RUNG_NAMES, build_networks, save_checkpoint
from constellai.models.marl.environment import SingleAgentAvoidanceEnv
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv

DEFAULT_ACTION_STD = 0.3  # fraction of the thrust limit


@dataclass
class TrainResult:
    networks: dict
    history: list
    extra_meta: dict = field(default_factory=dict)


def _multi_env(cfg: ScenarioConfig, fuel_weight: float) -> ScaledMultiAgentEnv:
    raw = MultiAgentAvoidanceEnv(
        n_agents=cfg.n_agents, mean_motion=cfg.mean_motion, dt=cfg.dt,
        collision_radius_m=cfg.collision_radius_m, max_thrust_mps2=cfg.max_thrust_mps2,
        max_steps=cfg.max_steps,
    )
    return ScaledMultiAgentEnv(raw, cfg.n_agents, cfg.max_thrust_mps2, cfg.dt, fuel_weight=fuel_weight)


def _single_env(cfg: ScenarioConfig, fuel_weight: float) -> ScaledSingleAgentEnv:
    raw = SingleAgentAvoidanceEnv(
        mean_motion=cfg.mean_motion, dt=cfg.dt, collision_radius_m=cfg.collision_radius_m,
        max_thrust_mps2=cfg.max_thrust_mps2, max_steps=cfg.max_steps,
    )
    return ScaledSingleAgentEnv(raw, cfg.max_thrust_mps2, cfg.dt, fuel_weight=fuel_weight)


def _set_action_std(networks: dict, action_std: float) -> None:
    # only the stochastic-policy networks have a log_std parameter
    for net in networks.values():
        if hasattr(net, "log_std"):
            net.log_std.data.fill_(math.log(action_std))


def _collision_rate(rows: list) -> float:
    return sum(r["any_collision"] for r in rows) / len(rows)


def _make_logger(rung: str, seed: int, log_every: int):
    started = time.time()

    def log(history: list) -> None:
        if log_every > 0 and len(history) % log_every == 0:
            recent = history[-log_every:]
            print(
                f"  [{rung} seed {seed}] episode {len(history)}: "
                f"trial collision rate (last {log_every}) {_collision_rate(recent):.2f}, "
                f"{time.time() - started:.0f}s elapsed",
                flush=True,
            )

    return log


def _ppo_row(episode: int, trajs: list, steps: int) -> dict:
    return {
        "episode": episode,
        "any_collision": any(t.collided for t in trajs),
        "agent_collision_fraction": sum(t.collided for t in trajs) / len(trajs),
        "episode_length": steps,
    }


def _train_reinforce(cfg, seed, episodes, lr, action_std, fuel_weight, log):
    single_cfg = replace(cfg, n_agents=1)
    networks = build_networks("reinforce", 1, POLICY_ACTION_LIMIT)
    _set_action_std(networks, action_std)
    policy, value = networks["policy"], networks["value"]
    env = _single_env(cfg, fuel_weight)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=lr)
    value_opt = torch.optim.Adam(value.parameters(), lr=lr)

    history = []
    for ep in range(episodes):
        positions, velocities = sample_scenario(make_rng(STREAM_TRAIN, seed, ep), single_cfg)
        traj = single_agent_baseline.collect_episode(env, policy, positions[0], velocities[0])
        policy_loss, value_loss = single_agent_baseline.reinforce_update(
            traj, policy, value, policy_opt, value_opt,
        )
        history.append({
            "episode": ep,
            "any_collision": bool(traj.collided),
            "agent_collision_fraction": float(traj.collided),
            "episode_length": len(traj.rewards),
            "policy_loss": policy_loss,
            "value_loss": value_loss,
        })
        log(history)
    return TrainResult(networks, history, {"penalty_coeff": single_agent_baseline.DEFAULT_PENALTY_COEFF})


def _train_ippo(cfg, seed, episodes, lr, action_std, fuel_weight, log):
    networks = build_networks("ippo", cfg.n_agents, POLICY_ACTION_LIMIT)
    _set_action_std(networks, action_std)
    policy = networks["policy"]
    env = _multi_env(cfg, fuel_weight)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    history = []
    for ep in range(episodes):
        positions, velocities = sample_scenario(make_rng(STREAM_TRAIN, seed, ep), cfg)
        trajs = ippo.collect_rollout(env, policy, positions, velocities)
        losses = ippo.ppo_update(trajs, policy, optimizer)
        row = _ppo_row(ep, trajs, max(len(t.rewards) for t in trajs))
        row.update(losses)
        history.append(row)
        log(history)
    return TrainResult(networks, history, {"penalty_coeff": ippo.DEFAULT_PENALTY_COEFF})


def _train_mappo(cfg, seed, episodes, lr, action_std, fuel_weight, log):
    networks = build_networks("mappo", cfg.n_agents, POLICY_ACTION_LIMIT)
    _set_action_std(networks, action_std)
    actor, critic = networks["actor"], networks["critic"]
    env = _multi_env(cfg, fuel_weight)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)

    history = []
    for ep in range(episodes):
        positions, velocities = sample_scenario(make_rng(STREAM_TRAIN, seed, ep), cfg)
        trajs = mappo.collect_rollout(env, actor, critic, positions, velocities)
        losses = mappo.mappo_update(trajs, actor, critic, actor_opt, critic_opt)
        row = _ppo_row(ep, trajs, max(len(t.rewards) for t in trajs))
        row.update(losses)
        history.append(row)
        log(history)
    return TrainResult(networks, history, {"penalty_coeff": mappo.DEFAULT_PENALTY_COEFF})


def _train_maddpg(cfg, seed, episodes, maddpg_lr, action_std, maddpg_updates, fuel_weight, log):
    networks = build_networks("maddpg", cfg.n_agents, POLICY_ACTION_LIMIT)
    actor, critic = networks["actor"], networks["critic"]
    target_actor, target_critic = copy.deepcopy(actor), copy.deepcopy(critic)
    env = _multi_env(cfg, fuel_weight)
    buffer = maddpg.ReplayBuffer()
    actor_opt = torch.optim.Adam(actor.parameters(), lr=maddpg_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=maddpg_lr)

    history = []
    for ep in range(episodes):
        positions, velocities = sample_scenario(make_rng(STREAM_TRAIN, seed, ep), cfg)
        any_collision = maddpg.collect_transitions(
            env, actor, positions, velocities, buffer, exploration_std=action_std,
        )
        last_losses = None
        for _ in range(maddpg_updates):
            out = maddpg.maddpg_update(
                buffer, cfg.n_agents, actor, critic, target_actor, target_critic, actor_opt, critic_opt,
            )
            if out is not None:
                last_losses = out
        row = {"episode": ep, "any_collision": bool(any_collision), "buffer_size": len(buffer)}
        if last_losses is not None:
            row.update(last_losses)
        history.append(row)
        log(history)
    return TrainResult(
        networks, history,
        {"penalty_coeff": maddpg.DEFAULT_PENALTY_COEFF, "updates_per_episode": maddpg_updates},
    )


def _train_constrained(cfg, seed, episodes, lr, action_std, initial_lambda, fuel_weight, log):
    networks = build_networks("constrained_mappo", cfg.n_agents, POLICY_ACTION_LIMIT)
    _set_action_std(networks, action_std)
    actor, reward_critic, cost_critic = networks["actor"], networks["reward_critic"], networks["cost_critic"]
    env = _multi_env(cfg, fuel_weight)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    reward_critic_opt = torch.optim.Adam(reward_critic.parameters(), lr=lr)
    cost_critic_opt = torch.optim.Adam(cost_critic.parameters(), lr=lr)

    lam = initial_lambda
    history = []
    for ep in range(episodes):
        positions, velocities = sample_scenario(make_rng(STREAM_TRAIN, seed, ep), cfg)
        trajs = constrained_mappo.collect_rollout(env, actor, reward_critic, cost_critic, positions, velocities)
        out = constrained_mappo.constrained_update(
            trajs, actor, reward_critic, cost_critic, lagrange_lambda=lam,
            actor_optimizer=actor_opt,
            reward_critic_optimizer=reward_critic_opt,
            cost_critic_optimizer=cost_critic_opt,
        )
        lam = out["lambda"]
        row = _ppo_row(ep, trajs, max(len(t.mission_rewards) for t in trajs))
        row.update(out)
        history.append(row)
        log(history)
    return TrainResult(
        networks, history,
        {
            "initial_lambda": initial_lambda,
            "final_lambda": lam,
            "cost_threshold_d": constrained_mappo.DEFAULT_COST_THRESHOLD_D,
            "lambda_lr": constrained_mappo.DEFAULT_LAMBDA_LR,
        },
    )


def train_rung(
    rung: str, cfg: ScenarioConfig, seed: int, episodes: int,
    lr: float = 5e-3, maddpg_lr: float = 1e-3, action_std: float = DEFAULT_ACTION_STD,
    maddpg_updates: int = 20, initial_lambda: float = 1.0, fuel_weight: float = DEFAULT_FUEL_WEIGHT,
    log_every: int = 50,
) -> TrainResult:
    if rung not in RUNG_NAMES:
        raise ValueError(f"unknown rung {rung!r}; expected one of {RUNG_NAMES}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    log = _make_logger(rung, seed, log_every)

    if rung == "reinforce":
        return _train_reinforce(cfg, seed, episodes, lr, action_std, fuel_weight, log)
    if rung == "ippo":
        return _train_ippo(cfg, seed, episodes, lr, action_std, fuel_weight, log)
    if rung == "mappo":
        return _train_mappo(cfg, seed, episodes, lr, action_std, fuel_weight, log)
    if rung == "maddpg":
        return _train_maddpg(cfg, seed, episodes, maddpg_lr, action_std, maddpg_updates, fuel_weight, log)
    return _train_constrained(cfg, seed, episodes, lr, action_std, initial_lambda, fuel_weight, log)


def _summary_line(rung: str, seed: int, history: list) -> str:
    window = max(1, min(100, len(history) // 2))
    first = _collision_rate(history[:window])
    last = _collision_rate(history[-window:])
    return f"[{rung} seed {seed}] trial collision rate: first {window} episodes {first:.2f} -> last {window} episodes {last:.2f}"


def _git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Train the M4 baseline ladder and save checkpoints")
    parser.add_argument("--rung", nargs="+", choices=[*RUNG_NAMES, "all"], default=["all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--p-collision-course", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--maddpg-lr", type=float, default=1e-3)
    parser.add_argument("--maddpg-updates", type=int, default=20, help="gradient updates per episode")
    parser.add_argument("--action-std", type=float, default=DEFAULT_ACTION_STD, help="exploration std, as a fraction of the thrust limit")
    parser.add_argument("--fuel-weight", type=float, default=DEFAULT_FUEL_WEIGHT, help="mission-reward penalty per step at a full burn")
    parser.add_argument("--initial-lambda", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--out-dir", default="outputs/m4_scaled")
    args = parser.parse_args()

    torch.set_num_threads(1)  # these networks are tiny; threading only adds overhead
    cfg = ScenarioConfig(n_agents=args.n_agents, p_collision_course=args.p_collision_course)
    rungs = list(RUNG_NAMES) if "all" in args.rung else args.rung
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = _git_commit()

    for rung in rungs:
        for seed in args.seeds:
            print(f"training {rung}, seed {seed}, {args.episodes} episodes", flush=True)
            started = time.time()
            result = train_rung(
                rung, cfg, seed, args.episodes, lr=args.lr, maddpg_lr=args.maddpg_lr,
                action_std=args.action_std, maddpg_updates=args.maddpg_updates,
                initial_lambda=args.initial_lambda, fuel_weight=args.fuel_weight,
                log_every=args.log_every,
            )
            meta = {
                "seed": seed,
                "episodes": args.episodes,
                "lr": args.maddpg_lr if rung == "maddpg" else args.lr,
                "action_std": args.action_std,
                "fuel_weight": args.fuel_weight,
                "scaled_env": True,
                "position_scale_m": POSITION_SCALE_M,
                "scenario_config": asdict(cfg),
                "git_commit": commit,
                "torch_version": torch.__version__,
                "train_seconds": time.time() - started,
                **result.extra_meta,
            }
            n_agents = 1 if rung == "reinforce" else cfg.n_agents
            # the checkpoint's limit field holds the POLICY-space action limit (1.0 = full thrust)
            save_checkpoint(out_dir / f"{rung}_seed{seed}.pt", rung, n_agents, POLICY_ACTION_LIMIT, result.networks, meta)
            (out_dir / f"{rung}_seed{seed}_history.json").write_text(json.dumps(result.history))
            print(_summary_line(rung, seed, result.history), flush=True)


if __name__ == "__main__":
    main()