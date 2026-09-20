"""M4 rung 4: MADDPG (Multi-Agent Deep Deterministic Policy Gradient).

The defining difference from MAPPO (rung 3): the centralized critic
here estimates Q(joint_observations, joint_ACTIONS) -- the value of a
specific joint action -- not V(joint_observations), the value of a
state under the current policy. This is precisely why MADDPG suits
continuous action spaces (Delta-v magnitude and direction, this
project's actual maneuver space) naturally: the deterministic actor is
trained by following the analytic gradient of this Q-function w.r.t.
its own action, rather than via a sampled, scored stochastic policy
(PPO's approach in rungs 2-3).

Off-policy, not on-policy: uses a replay buffer and target networks
with soft (Polyak) updates, standard DDPG/MADDPG machinery for
training stability -- this is also a structural difference from
IPPO/MAPPO's on-policy rollout-then-discard approach, not just a
different network shape.

DELIBERATE deviation from the original MADDPG paper: that paper trains
a SEPARATE policy per agent, since it targets mixed cooperative-
competitive settings with potentially different per-agent objectives.
These satellites are homogeneous with an identical objective, so
parameter sharing is kept here -- the same convention already used for
IPPO and MAPPO -- so this rung's comparison isolates "what changed
algorithmically" without also varying "shared vs separate weights" as
a second, confounding axis.
"""

from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv

DEFAULT_PENALTY_COEFF = 50.0
DEFAULT_GAMMA = 0.99
DEFAULT_TAU = 0.01  # Polyak soft-update rate
DEFAULT_EXPLORATION_STD = 0.1
DEFAULT_BUFFER_SIZE = 10_000
DEFAULT_BATCH_SIZE = 64


class DeterministicActor(nn.Module):
    """Local-observation-only, outputs a single deterministic action
    (not a distribution) -- shared across all agents, per the module
    docstring's disclosed deviation from the original paper."""

    def __init__(self, obs_dim: int = 7, action_dim: int = 3, hidden_dim: int = 32, max_action: float = 0.01):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs) * self.max_action


class CentralizedQCritic(nn.Module):
    """Q(joint_obs, joint_actions, agent_id) -- the actual MADDPG-
    defining input: joint ACTIONS are part of the input, not just
    joint observations (see module docstring)."""

    def __init__(self, n_agents: int, obs_dim: int = 7, action_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.n_agents = n_agents
        input_dim = n_agents * obs_dim + n_agents * action_dim + n_agents
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_obs: torch.Tensor, joint_actions: torch.Tensor, agent_id: int) -> torch.Tensor:
        one_hot = torch.zeros(self.n_agents)
        one_hot[agent_id] = 1.0
        x = torch.cat([joint_obs, joint_actions, one_hot], dim=-1)
        return self.net(x).squeeze(-1)


@dataclass
class Transition:
    obs: list  # per-agent local obs, length n_agents
    actions: list
    rewards: list
    next_obs: list
    dones: list


class ReplayBuffer:
    def __init__(self, capacity: int = DEFAULT_BUFFER_SIZE):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


def collect_transitions(
    env: MultiAgentAvoidanceEnv,
    actor: DeterministicActor,
    initial_positions_m: list[np.ndarray],
    initial_velocities_mps: list[np.ndarray],
    buffer: ReplayBuffer,
    penalty_coeff: float = DEFAULT_PENALTY_COEFF,
    exploration_std: float = DEFAULT_EXPLORATION_STD,
) -> bool:
    """Runs one full episode, pushing every transition into the replay
    buffer as it goes (off-policy: data collection and training are
    decoupled, unlike IPPO/MAPPO's collect-then-immediately-train
    loop). Returns whether any agent collided this episode."""
    n_agents = env.n_agents
    obs_list = env.reset(initial_positions_m, initial_velocities_mps)
    any_collision = False
    done_flags = [False] * n_agents

    while not all(done_flags):
        actions = []
        with torch.no_grad():
            for i in range(n_agents):
                obs = torch.from_numpy(obs_list[i]).float()
                action = actor(obs).numpy()
                noise = np.random.normal(0, exploration_std, size=action.shape)
                actions.append(action + noise)

        next_obs_list, result = env.step(actions)

        rewards = [
            result.mission_rewards[i] - penalty_coeff * result.safety_costs[i]
            for i in range(n_agents)
        ]
        if any(result.infos[i].get("collided") for i in range(n_agents)):
            any_collision = True

        buffer.push(Transition(obs_list, actions, rewards, next_obs_list, result.dones))

        obs_list = next_obs_list
        done_flags = result.dones

    return any_collision


def maddpg_update(
    buffer: ReplayBuffer,
    n_agents: int,
    actor: DeterministicActor,
    critic: CentralizedQCritic,
    target_actor: DeterministicActor,
    target_critic: CentralizedQCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    batch_size: int = DEFAULT_BATCH_SIZE,
    gamma: float = DEFAULT_GAMMA,
    tau: float = DEFAULT_TAU,
) -> dict[str, float] | None:
    """One off-policy update step, sampled from the replay buffer.
    Returns None if the buffer doesn't have enough transitions yet.

    Critic loss: standard TD/Bellman target using TARGET networks (not
    the live ones) -- this is why target networks with soft updates
    exist at all: bootstrapping against a live, rapidly-changing
    network is a well-known source of training instability in
    off-policy deep RL.

    Actor loss: the deterministic policy gradient -- maximize
    Q(joint_obs, [live_actor(own_obs), other_agents'_actions_from_buffer])
    w.r.t. the actor's own parameters. This is what "following the
    Q-function's gradient" concretely means in code.
    """
    if len(buffer) < batch_size:
        return None

    batch = buffer.sample(batch_size)

    critic_losses, actor_losses = [], []

    for agent_id in range(n_agents):
        joint_obs = torch.stack([
            torch.from_numpy(np.concatenate(t.obs)).float() for t in batch
        ])
        joint_actions = torch.stack([
            torch.from_numpy(np.concatenate(t.actions)).float() for t in batch
        ])
        joint_next_obs = torch.stack([
            torch.from_numpy(np.concatenate(t.next_obs)).float() for t in batch
        ])
        rewards = torch.tensor([t.rewards[agent_id] for t in batch], dtype=torch.float32)
        dones = torch.tensor([float(t.dones[agent_id]) for t in batch], dtype=torch.float32)

        # --- Critic update ---
        with torch.no_grad():
            next_actions_per_agent = []
            for i in range(n_agents):
                next_obs_i = torch.stack([torch.from_numpy(t.next_obs[i]).float() for t in batch])
                next_actions_per_agent.append(target_actor(next_obs_i))
            joint_next_actions = torch.cat(next_actions_per_agent, dim=-1)

            target_q = torch.stack([
                target_critic(joint_next_obs[j], joint_next_actions[j], agent_id)
                for j in range(len(batch))
            ])
            td_target = rewards + gamma * (1 - dones) * target_q

        current_q = torch.stack([
            critic(joint_obs[j], joint_actions[j], agent_id) for j in range(len(batch))
        ])
        critic_loss = nn.functional.mse_loss(current_q, td_target)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # --- Actor update (deterministic policy gradient) ---
        own_obs = torch.stack([torch.from_numpy(t.obs[agent_id]).float() for t in batch])
        predicted_own_action = actor(own_obs)

        # Rebuild joint action with THIS agent's action replaced by the
        # live actor's output (gradient flows through here), others held
        # fixed from the buffer -- standard MADDPG actor-update pattern.
        joint_actions_for_actor_update = joint_actions.clone()
        start = agent_id * predicted_own_action.shape[-1]
        end = start + predicted_own_action.shape[-1]
        joint_actions_for_actor_update = torch.cat([
            joint_actions_for_actor_update[:, :start],
            predicted_own_action,
            joint_actions_for_actor_update[:, end:],
        ], dim=-1)

        actor_loss = -torch.stack([
            critic(joint_obs[j], joint_actions_for_actor_update[j], agent_id)
            for j in range(len(batch))
        ]).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        critic_losses.append(critic_loss.item())
        actor_losses.append(actor_loss.item())

    soft_update(target_actor, actor, tau)
    soft_update(target_critic, critic, tau)

    return {
        "critic_loss": sum(critic_losses) / n_agents,
        "actor_loss": sum(actor_losses) / n_agents,
    }


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Polyak averaging: target = tau * source + (1 - tau) * target.
    Small tau -> slow, stable target-network tracking -- the mechanism
    that prevents the moving-target instability off-policy bootstrapping
    is otherwise prone to."""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)