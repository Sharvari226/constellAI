"""M4 rung 5: Constrained/Lagrangian MAPPO -- the actual proposed
method this project's whole architecture argues for.

Every prior rung (1-4) deliberately combined mission_reward and
safety_cost into ONE scalar via a fixed penalty_coeff, precisely so
this project would have an honest reward-penalty baseline to compare
against -- that combination is the documented CPO anti-pattern (a fixed
penalty either lets the policy ignore safety, or causes it to abandon
the mission, with no stable middle ground, because a fixed number
can't adapt to how close the CURRENT policy is to violating the
constraint).

This rung is the only one that keeps mission_reward and safety_cost
separate through the entire training loop:
  - TWO critics: one estimates expected discounted mission return
    (same as MAPPO's), one estimates expected discounted COST return
    (a new addition -- "how much collision-cost do I expect to
    accumulate under my current policy").
  - ONE Lagrange multiplier (lambda), updated via dual ascent:
        lambda <- max(0, lambda + lr * (measured_cost_return - d))
    where d is the target cost threshold (the actual "P(collision) <=
    epsilon" constraint, in per-episode-cost-return terms). If the
    policy is currently violating the constraint (cost_return > d),
    lambda grows, making the policy update weight safety more heavily.
    If the policy is comfortably safe (cost_return < d), lambda
    shrinks toward 0, letting the policy focus more on the mission --
    this automatic adaptation is precisely what a fixed penalty_coeff
    cannot do.
  - The actor's PPO objective uses a COMBINED advantage,
    adv = adv_reward - lambda * adv_cost, then applies the exact same
    clipped-surrogate objective as every prior PPO-based rung. This
    combined-advantage form is the standard, simplified Lagrangian-PPO
    pattern (as used in e.g. PPO-Lagrangian implementations) -- it
    achieves the CMDP's "maximize reward subject to bounded cost"
    behavior without needing a full constrained-optimization solver
    (CPO's own trust-region approach) at this "lite" scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from constellai.models.marl.ippo import compute_gae
from constellai.models.marl.mappo import Actor, CentralizedCritic
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv

DEFAULT_COST_THRESHOLD_D = 0.05  # target: average per-episode cost-return should stay under this
DEFAULT_LAMBDA_LR = 0.01
DEFAULT_CLIP_EPS = 0.2
DEFAULT_PPO_EPOCHS = 4
DEFAULT_ENTROPY_COEFF = 0.01


@dataclass
class AgentTrajectory:
    observations: list = field(default_factory=list)
    joint_observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    reward_values: list = field(default_factory=list)   # from the reward critic
    cost_values: list = field(default_factory=list)      # from the cost critic -- NEW vs MAPPO
    mission_rewards: list = field(default_factory=list)  # kept SEPARATE from safety_costs -- the whole point
    safety_costs: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    collided: bool = False


def collect_rollout(
    env: MultiAgentAvoidanceEnv,
    actor: Actor,
    reward_critic: CentralizedCritic,
    cost_critic: CentralizedCritic,
    initial_positions_m: list[np.ndarray],
    initial_velocities_mps: list[np.ndarray],
) -> list[AgentTrajectory]:
    """Same structure as MAPPO's rollout, but NEVER combines
    mission_reward and safety_cost -- they're recorded, valued, and
    later advantage-estimated completely independently."""
    n_agents = env.n_agents
    trajs = [AgentTrajectory() for _ in range(n_agents)]

    obs_list = env.reset(initial_positions_m, initial_velocities_mps)
    all_done = False

    while not all_done:
        joint_obs = torch.from_numpy(np.concatenate(obs_list)).float()
        actions = []
        for i in range(n_agents):
            obs = torch.from_numpy(obs_list[i]).float()
            dist = actor(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            reward_value = reward_critic(joint_obs, agent_id=i)
            cost_value = cost_critic(joint_obs, agent_id=i)

            actions.append(action.detach().numpy())
            trajs[i].observations.append(obs)
            trajs[i].joint_observations.append(joint_obs)
            trajs[i].actions.append(action)
            trajs[i].log_probs.append(log_prob.detach())
            trajs[i].reward_values.append(reward_value.detach())
            trajs[i].cost_values.append(cost_value.detach())

        obs_list, result = env.step(actions)

        for i in range(n_agents):
            trajs[i].mission_rewards.append(result.mission_rewards[i])
            trajs[i].safety_costs.append(result.safety_costs[i])
            trajs[i].dones.append(result.dones[i])
            if result.infos[i].get("collided"):
                trajs[i].collided = True

        all_done = all(result.dones)

    return trajs


def constrained_update(
    trajs: list[AgentTrajectory],
    actor: Actor,
    reward_critic: CentralizedCritic,
    cost_critic: CentralizedCritic,
    lagrange_lambda: float,
    actor_optimizer: torch.optim.Optimizer,
    reward_critic_optimizer: torch.optim.Optimizer,
    cost_critic_optimizer: torch.optim.Optimizer,
    cost_threshold_d: float = DEFAULT_COST_THRESHOLD_D,
    lambda_lr: float = DEFAULT_LAMBDA_LR,
    clip_eps: float = DEFAULT_CLIP_EPS,
    epochs: int = DEFAULT_PPO_EPOCHS,
    entropy_coeff: float = DEFAULT_ENTROPY_COEFF,
) -> dict[str, float]:
    """One constrained-PPO update. Returns the UPDATED lambda alongside
    the usual losses -- lambda is not a module parameter, it's returned
    and threaded through by the caller, since dual ascent on it is
    conceptually separate from gradient descent on the networks."""
    all_obs, all_joint_obs, all_agent_ids = [], [], []
    all_actions, all_old_log_probs = [], []
    all_reward_advantages, all_reward_returns = [], []
    all_cost_advantages, all_cost_returns = [], []

    for i, traj in enumerate(trajs):
        if not traj.mission_rewards:
            continue
        r_adv, r_ret = compute_gae(traj.mission_rewards, [v.item() for v in traj.reward_values], traj.dones)
        c_adv, c_ret = compute_gae(traj.safety_costs, [v.item() for v in traj.cost_values], traj.dones)

        all_obs.extend(traj.observations)
        all_joint_obs.extend(traj.joint_observations)
        all_agent_ids.extend([i] * len(traj.mission_rewards))
        all_actions.extend(traj.actions)
        all_old_log_probs.extend(traj.log_probs)
        all_reward_advantages.append(r_adv)
        all_reward_returns.append(r_ret)
        all_cost_advantages.append(c_adv)
        all_cost_returns.append(c_ret)

    obs_batch = torch.stack(all_obs)
    joint_obs_batch = torch.stack(all_joint_obs)
    actions_batch = torch.stack(all_actions)
    old_log_probs_batch = torch.stack(all_old_log_probs)

    reward_adv_batch = torch.cat(all_reward_advantages)
    reward_ret_batch = torch.cat(all_reward_returns)
    cost_adv_batch = torch.cat(all_cost_advantages)
    cost_ret_batch = torch.cat(all_cost_returns)

    reward_adv_batch = (reward_adv_batch - reward_adv_batch.mean()) / (reward_adv_batch.std() + 1e-8)
    cost_adv_batch = (cost_adv_batch - cost_adv_batch.mean()) / (cost_adv_batch.std() + 1e-8)

    # THE Lagrangian combination -- reward advantage minus lambda times
    # cost advantage. This single line is where "safety as a hard
    # constraint, adapted automatically" concretely happens, replacing
    # every prior rung's fixed "reward - penalty_coeff * cost".
    combined_advantage = reward_adv_batch - lagrange_lambda * cost_adv_batch

    policy_losses, reward_value_losses, cost_value_losses = [], [], []
    for _ in range(epochs):
        dist = actor(obs_batch)
        new_log_probs = dist.log_prob(actions_batch).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        ratio = torch.exp(new_log_probs - old_log_probs_batch)
        surr1 = ratio * combined_advantage
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * combined_advantage
        policy_loss = -torch.min(surr1, surr2).mean() - entropy_coeff * entropy

        actor_optimizer.zero_grad()
        policy_loss.backward()
        actor_optimizer.step()

        reward_values = torch.stack([
            reward_critic(joint_obs_batch[j], agent_id=all_agent_ids[j]) for j in range(len(all_agent_ids))
        ])
        reward_value_loss = nn.functional.mse_loss(reward_values, reward_ret_batch)
        reward_critic_optimizer.zero_grad()
        reward_value_loss.backward()
        reward_critic_optimizer.step()

        cost_values = torch.stack([
            cost_critic(joint_obs_batch[j], agent_id=all_agent_ids[j]) for j in range(len(all_agent_ids))
        ])
        cost_value_loss = nn.functional.mse_loss(cost_values, cost_ret_batch)
        cost_critic_optimizer.zero_grad()
        cost_value_loss.backward()
        cost_critic_optimizer.step()

        policy_losses.append(policy_loss.item())
        reward_value_losses.append(reward_value_loss.item())
        cost_value_losses.append(cost_value_loss.item())

    # Dual ascent on lambda: if measured cost return exceeds the
    # threshold, lambda grows (weight safety more next update); if
    # comfortably under, lambda shrinks toward 0 but never negative
    # (a negative lambda would mean "rewarding" unsafe behavior, which
    # is never correct).
    mean_cost_return = cost_ret_batch.mean().item()
    new_lambda = max(0.0, lagrange_lambda + lambda_lr * (mean_cost_return - cost_threshold_d))

    return {
        "policy_loss": sum(policy_losses) / epochs,
        "reward_value_loss": sum(reward_value_losses) / epochs,
        "cost_value_loss": sum(cost_value_losses) / epochs,
        "mean_cost_return": mean_cost_return,
        "lambda": new_lambda,
    }