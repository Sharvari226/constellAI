"""M4 baseline rung 1: single-agent RL, no coordination, no explicit
safety constraint -- echoes Bourriez et al.'s single-spacecraft POMDP
formulation. This is the floor the rest of the ladder (IPPO, MAPPO,
MADDPG, constrained) has to beat.

REINFORCE with a value-function baseline (for variance reduction) --
the simplest correct policy-gradient method, appropriate for a first
rung whose entire purpose is being a simple, well-understood floor, not
a sophisticated method in its own right.

This is the ONE place in the ladder where mission_reward and
safety_cost get combined into a single scalar (reward = mission_reward
- penalty_coeff * safety_cost) -- everywhere else in this project,
that combination is the exact anti-pattern being avoided (see CPO's
fixed-penalty experiment, already in the research spec). It's
deliberate and disclosed here: this rung EXISTS to represent the naive,
unconstrained approach the constrained rungs get compared against.
Combining reward and safety here is not a design slip, it's the point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from constellai.models.marl.environment import SingleAgentAvoidanceEnv

DEFAULT_PENALTY_COEFF = 50.0  # placeholder, not yet spec-frozen -- see module docstring


class GaussianPolicy(nn.Module):
    """State (7,) -> Gaussian thrust distribution over R^3.

    log_std is a learned parameter, not state-dependent -- standard
    simplification for a first baseline; state-dependent variance is a
    reasonable later improvement, not required for this rung to be a
    valid floor.
    """

    def __init__(self, obs_dim: int = 7, action_dim: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)  # starts near std=0.37

    def forward(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std).clamp(min=1e-3)
        return torch.distributions.Normal(mean, std)


class ValueNet(nn.Module):
    """State (7,) -> scalar value estimate, used only as a REINFORCE
    baseline for variance reduction -- not part of the policy itself."""

    def __init__(self, obs_dim: int = 7, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


@dataclass
class Trajectory:
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    safety_costs: list = field(default_factory=list)
    collided: bool = False


def collect_episode(
    env: SingleAgentAvoidanceEnv,
    policy: GaussianPolicy,
    initial_position_m: np.ndarray,
    initial_velocity_mps: np.ndarray,
    penalty_coeff: float = DEFAULT_PENALTY_COEFF,
) -> Trajectory:
    """Runs one full episode under the current policy, returning
    everything needed for a REINFORCE update."""
    traj = Trajectory()
    obs_np = env.reset(initial_position_m, initial_velocity_mps)

    done = False
    while not done:
        obs = torch.from_numpy(obs_np).float()
        dist = policy(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        obs_np, result = env.step(action.detach().numpy())

        combined_reward = result.mission_reward - penalty_coeff * result.safety_cost

        traj.observations.append(obs)
        traj.actions.append(action)
        traj.log_probs.append(log_prob)
        traj.rewards.append(combined_reward)
        traj.safety_costs.append(result.safety_cost)
        if result.info["collided"]:
            traj.collided = True

        done = result.done

    return traj


def compute_discounted_returns(rewards: list[float], gamma: float = 0.99) -> torch.Tensor:
    """Standard discounted return-to-go, one value per timestep."""
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.insert(0, running)
    return torch.tensor(returns, dtype=torch.float32)


def reinforce_update(
    traj: Trajectory,
    policy: GaussianPolicy,
    value_net: ValueNet,
    policy_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    gamma: float = 0.99,
) -> tuple[float, float]:
    """One REINFORCE-with-baseline update from a single trajectory.

    Returns
    -------
    (policy_loss, value_loss) : tuple[float, float]
    """
    returns = compute_discounted_returns(traj.rewards, gamma=gamma)
    obs_batch = torch.stack(traj.observations)

    values = value_net(obs_batch)
    advantages = (returns - values.detach())
    # Standardize advantages -- standard variance-reduction practice,
    # not optional for this to train stably.
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    log_probs = torch.stack(traj.log_probs)
    policy_loss = -(log_probs * advantages).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    value_loss = nn.functional.mse_loss(values, returns)
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    return policy_loss.item(), value_loss.item()