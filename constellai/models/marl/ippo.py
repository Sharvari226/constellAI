"""M4 rung 2: IPPO (Independent PPO) -- N agents, shared network
parameters (they're physically identical satellites, so sharing is the
sample-efficient choice, not a shortcut), but each agent's policy AND
value estimate depend ONLY on its own local observation. No agent ever
sees another agent's state, during training or execution -- that is
the actual definition of "independent" here, not whether weights are
shared. This is what distinguishes IPPO from MAPPO (rung 3): MAPPO adds
a centralized critic with access to joint/global state during training;
IPPO's critic is exactly as decentralized as its actor.

This rung also upgrades the optimization algorithm itself relative to
rung 1's plain REINFORCE: PPO's clipped surrogate objective, Generalized
Advantage Estimation (GAE), and multiple gradient epochs per collected
batch. That's a separate, real change worth keeping distinct from "more
agents" when interpreting any performance difference from rung 1 --
if IPPO outperforms rung 1, part of that could be the better
optimizer, not coordination-related at all. Isolating those two
effects properly is future ablation work, not solved by this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv

DEFAULT_PENALTY_COEFF = 50.0  # same placeholder status as rung 1 -- not yet spec-frozen
DEFAULT_CLIP_EPS = 0.2
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_GAMMA = 0.99
DEFAULT_PPO_EPOCHS = 4
DEFAULT_ENTROPY_COEFF = 0.01


class ActorCritic(nn.Module):
    """Shared trunk, split into a Gaussian policy head and a scalar
    value head. One instance of this IS the "shared parameters across
    homogeneous agents" -- every agent calls the SAME instance."""

    def __init__(self, obs_dim: int = 7, action_dim: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh())
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor]:
        features = self.trunk(obs)
        mean = self.mean_head(features)
        std = torch.exp(self.log_std).clamp(min=1e-3)
        dist = torch.distributions.Normal(mean, std)
        value = self.value_head(features).squeeze(-1)
        return dist, value


@dataclass
class AgentTrajectory:
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    collided: bool = False


def collect_rollout(
    env: MultiAgentAvoidanceEnv,
    policy: ActorCritic,
    initial_positions_m: list[np.ndarray],
    initial_velocities_mps: list[np.ndarray],
    penalty_coeff: float = DEFAULT_PENALTY_COEFF,
) -> list[AgentTrajectory]:
    """Runs one episode, returning one AgentTrajectory PER agent -- since
    parameters are shared, all N trajectories get pooled into one PPO
    update afterward (this is exactly the sample-efficiency benefit of
    sharing: N agents' experience trains one set of weights)."""
    n_agents = env.n_agents
    trajs = [AgentTrajectory() for _ in range(n_agents)]

    obs_list = env.reset(initial_positions_m, initial_velocities_mps)
    all_done = False

    while not all_done:
        actions = []
        for i in range(n_agents):
            obs = torch.from_numpy(obs_list[i]).float()
            dist, value = policy(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            actions.append(action.detach().numpy())
            trajs[i].observations.append(obs)
            trajs[i].actions.append(action)
            trajs[i].log_probs.append(log_prob.detach())
            trajs[i].values.append(value.detach())

        obs_list, result = env.step(actions)

        for i in range(n_agents):
            combined_reward = result.mission_rewards[i] - penalty_coeff * result.safety_costs[i]
            trajs[i].rewards.append(combined_reward)
            trajs[i].dones.append(result.dones[i])
            if result.infos[i].get("collided"):
                trajs[i].collided = True

        all_done = all(result.dones)

    return trajs


def compute_gae(
    rewards: list[float], values: list[float], dones: list[bool],
    gamma: float = DEFAULT_GAMMA, lam: float = DEFAULT_GAE_LAMBDA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation. Returns (advantages, returns),
    both length len(rewards). Standard GAE recursion; bootstrap value
    after the last step is 0 (episode genuinely ends here, not
    truncated) -- correct for this environment since done always means
    collision, fuel exhaustion, or max_steps, never an artificial cutoff.
    """
    advantages = [0.0] * len(rewards)
    last_gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        next_non_terminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae
        next_value = values[t]

    advantages_t = torch.tensor(advantages, dtype=torch.float32)
    values_t = torch.tensor(values, dtype=torch.float32)
    returns_t = advantages_t + values_t
    return advantages_t, returns_t


def ppo_update(
    trajs: list[AgentTrajectory],
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    clip_eps: float = DEFAULT_CLIP_EPS,
    epochs: int = DEFAULT_PPO_EPOCHS,
    entropy_coeff: float = DEFAULT_ENTROPY_COEFF,
) -> dict[str, float]:
    """Pools ALL agents' trajectories into one batch (this is the
    parameter-sharing payoff) and runs `epochs` passes of the PPO
    clipped-surrogate update over that pooled batch."""
    all_obs, all_actions, all_old_log_probs, all_advantages, all_returns = [], [], [], [], []

    for traj in trajs:
        if not traj.rewards:
            continue
        advantages, returns = compute_gae(traj.rewards, [v.item() for v in traj.values], traj.dones)
        all_obs.extend(traj.observations)
        all_actions.extend(traj.actions)
        all_old_log_probs.extend(traj.log_probs)
        all_advantages.append(advantages)
        all_returns.append(returns)

    obs_batch = torch.stack(all_obs)
    actions_batch = torch.stack(all_actions)
    old_log_probs_batch = torch.stack(all_old_log_probs)
    advantages_batch = torch.cat(all_advantages)
    returns_batch = torch.cat(all_returns)

    # Standardize advantages across the WHOLE pooled batch -- consistent
    # with rung 1's per-episode standardization, now done once across
    # all agents' experience together, since that's the batch PPO
    # actually trains on.
    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

    policy_losses, value_losses = [], []
    for _ in range(epochs):
        dist, values = policy(obs_batch)
        new_log_probs = dist.log_prob(actions_batch).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        ratio = torch.exp(new_log_probs - old_log_probs_batch)
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_batch
        policy_loss = -torch.min(surr1, surr2).mean() - entropy_coeff * entropy

        value_loss = nn.functional.mse_loss(values, returns_batch)

        loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())

    return {"policy_loss": sum(policy_losses) / epochs, "value_loss": sum(value_losses) / epochs}