"""M4 rung 3: MAPPO (Multi-Agent PPO) -- Centralized Training,
Decentralized Execution (CTDE).

The actor is IDENTICAL in structure to IPPO's: local observation only,
because a real satellite cannot see every other satellite's exact
state at execution time -- decentralized execution is not optional,
it's physically required. What changes from IPPO is the CRITIC: during
training only, it receives the concatenation of every agent's
observation (the "centralized"/joint state) plus which agent's value
it is currently estimating, rather than being restricted to that one
agent's local observation. This is the actual, precise distinction
this rung exists to test -- does access to joint state during training
alone (with execution still fully decentralized) improve coordination
over IPPO's fully-decentralized critic?

Disclosed limitation, not silently ignored: the centralized critic's
input dimension is n_agents * obs_dim + n_agents (one-hot agent id) --
FIXED at training time. This does not natively answer "what happens at
N=1000 satellites" -- it is exactly the same category of scaling
question the SafeRL survey's P3 (decentralized SafeMARL) open problem
raises, and exactly why M2's sparse graph exists on the forecasting
side. MAPPO at this rung is a valid baseline-ladder comparison point,
not a scalability solution -- that gap is real and stays open for
later (constrained/graph-informed rungs) to address, not this one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from constellai.models.marl.ippo import compute_gae
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv

DEFAULT_PENALTY_COEFF = 50.0
DEFAULT_CLIP_EPS = 0.2
DEFAULT_PPO_EPOCHS = 4
DEFAULT_ENTROPY_COEFF = 0.01


class Actor(nn.Module):
    """Local-observation-only policy -- structurally identical to
    IPPO's actor half. Decentralized execution requires this to never
    take anything but obs_dim as input, ever."""

    def __init__(self, obs_dim: int = 7, action_dim: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh())
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)

    def forward(self, obs: torch.Tensor) -> torch.distributions.Normal:
        features = self.net(obs)
        mean = self.mean_head(features)
        std = torch.exp(self.log_std).clamp(min=1e-3)
        return torch.distributions.Normal(mean, std)


class CentralizedCritic(nn.Module):
    """Input: concat(all agents' local observations) + one-hot(which
    agent's value this is) -- the "centralized" half of CTDE. This
    module is used ONLY during training; nothing about execution ever
    calls it."""

    def __init__(self, n_agents: int, obs_dim: int = 7, hidden_dim: int = 64):
        super().__init__()
        self.n_agents = n_agents
        input_dim = n_agents * obs_dim + n_agents
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_obs: torch.Tensor, agent_id: int) -> torch.Tensor:
        """joint_obs: shape (n_agents * obs_dim,) -- all agents'
        observations concatenated in a fixed, consistent order."""
        one_hot = torch.zeros(self.n_agents)
        one_hot[agent_id] = 1.0
        critic_input = torch.cat([joint_obs, one_hot])
        return self.net(critic_input).squeeze(-1)


@dataclass
class AgentTrajectory:
    observations: list = field(default_factory=list)
    joint_observations: list = field(default_factory=list)  # NEW vs IPPO: needed for the centralized critic
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    collided: bool = False


def collect_rollout(
    env: MultiAgentAvoidanceEnv,
    actor: Actor,
    critic: CentralizedCritic,
    initial_positions_m: list[np.ndarray],
    initial_velocities_mps: list[np.ndarray],
    penalty_coeff: float = DEFAULT_PENALTY_COEFF,
) -> list[AgentTrajectory]:
    """Same rollout structure as IPPO's, but ALSO records the joint
    observation at each timestep (needed for the centralized critic)
    and queries the critic with each agent's id, not just its own obs."""
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
            value = critic(joint_obs, agent_id=i)

            actions.append(action.detach().numpy())
            trajs[i].observations.append(obs)
            trajs[i].joint_observations.append(joint_obs)
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


def mappo_update(
    trajs: list[AgentTrajectory],
    actor: Actor,
    critic: CentralizedCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    clip_eps: float = DEFAULT_CLIP_EPS,
    epochs: int = DEFAULT_PPO_EPOCHS,
    entropy_coeff: float = DEFAULT_ENTROPY_COEFF,
) -> dict[str, float]:
    """Actor update is the SAME clipped-surrogate PPO objective as
    IPPO's -- the only difference this rung introduces is where the
    advantage estimate (via GAE) comes from: the centralized critic's
    values, not a decentralized one. Critic is retrained via MSE
    against GAE returns, same as IPPO's value head, just with joint-
    state + agent-id input instead of local-obs-only input."""
    all_obs, all_joint_obs, all_agent_ids = [], [], []
    all_actions, all_old_log_probs, all_advantages, all_returns = [], [], [], []

    for i, traj in enumerate(trajs):
        if not traj.rewards:
            continue
        advantages, returns = compute_gae(traj.rewards, [v.item() for v in traj.values], traj.dones)
        all_obs.extend(traj.observations)
        all_joint_obs.extend(traj.joint_observations)
        all_agent_ids.extend([i] * len(traj.rewards))
        all_actions.extend(traj.actions)
        all_old_log_probs.extend(traj.log_probs)
        all_advantages.append(advantages)
        all_returns.append(returns)

    obs_batch = torch.stack(all_obs)
    joint_obs_batch = torch.stack(all_joint_obs)
    actions_batch = torch.stack(all_actions)
    old_log_probs_batch = torch.stack(all_old_log_probs)
    advantages_batch = torch.cat(all_advantages)
    returns_batch = torch.cat(all_returns)

    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

    policy_losses, value_losses = [], []
    for _ in range(epochs):
        dist = actor(obs_batch)
        new_log_probs = dist.log_prob(actions_batch).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        ratio = torch.exp(new_log_probs - old_log_probs_batch)
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_batch
        policy_loss = -torch.min(surr1, surr2).mean() - entropy_coeff * entropy

        actor_optimizer.zero_grad()
        policy_loss.backward()
        actor_optimizer.step()

        values = torch.stack([
            critic(joint_obs_batch[j], agent_id=all_agent_ids[j])
            for j in range(len(all_agent_ids))
        ])
        value_loss = nn.functional.mse_loss(values, returns_batch)

        critic_optimizer.zero_grad()
        value_loss.backward()
        critic_optimizer.step()

        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())

    return {"policy_loss": sum(policy_losses) / epochs, "value_loss": sum(value_losses) / epochs}