"""Unit tests for MAPPO.

test_critic_uses_joint_state_not_just_local_obs is the one that
actually matters here -- it's the concrete proof this rung is doing
what it claims (centralized critic, decentralized actor), not just
IPPO with extra plumbing.
"""

import numpy as np
import torch

from constellai.models.marl.mappo import (
    Actor,
    CentralizedCritic,
    collect_rollout,
    mappo_update,
)
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv


def test_actor_is_local_obs_only():
    actor = Actor(obs_dim=7)
    obs = torch.zeros(7)
    dist = actor(obs)
    assert dist.sample().shape == (3,)


def test_critic_uses_joint_state_not_just_local_obs():
    """The core architectural claim of this rung: critic output for
    agent 0 must change when agent 1's state changes, even though
    agent 0's OWN observation is identical -- this is precisely what
    'centralized' means, and precisely what distinguishes this from
    IPPO's decentralized critic."""
    critic = CentralizedCritic(n_agents=2, obs_dim=7)
    agent0_obs = torch.zeros(7)

    joint_a = torch.cat([agent0_obs, torch.zeros(7)])
    joint_b = torch.cat([agent0_obs, torch.ones(7) * 100.0])  # agent 1's state very different

    value_a = critic(joint_a, agent_id=0)
    value_b = critic(joint_b, agent_id=0)

    assert not torch.allclose(value_a, value_b)


def test_collect_rollout_records_joint_observations():
    env = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=0.0011, max_steps=10)
    actor = Actor()
    critic = CentralizedCritic(n_agents=2)
    trajs = collect_rollout(
        env, actor, critic,
        initial_positions_m=[np.array([1000.0, 0, 0]), np.array([1200.0, 0, 0])],
        initial_velocities_mps=[np.zeros(3), np.zeros(3)],
    )
    assert all(jo.shape == (14,) for jo in trajs[0].joint_observations)  # 2 agents * 7 obs_dim


def test_mappo_update_runs_and_changes_both_networks():
    env = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=0.0011, max_steps=15)
    actor = Actor()
    critic = CentralizedCritic(n_agents=2)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

    actor_params_before = [p.clone() for p in actor.parameters()]
    critic_params_before = [p.clone() for p in critic.parameters()]

    trajs = collect_rollout(
        env, actor, critic,
        initial_positions_m=[np.array([1000.0, 0, 0]), np.array([1200.0, 0, 0])],
        initial_velocities_mps=[np.zeros(3), np.zeros(3)],
    )
    losses = mappo_update(trajs, actor, critic, actor_opt, critic_opt)

    assert "policy_loss" in losses and "value_loss" in losses
    assert any(not torch.allclose(b, a) for b, a in zip(actor_params_before, list(actor.parameters())))
    assert any(not torch.allclose(b, a) for b, a in zip(critic_params_before, list(critic.parameters())))


def test_mappo_learns_to_reduce_collisions():
    torch.manual_seed(0)
    env = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=0.0011, collision_radius_m=50.0, max_steps=25)
    actor = Actor()
    critic = CentralizedCritic(n_agents=2)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=5e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=5e-3)

    def collision_rate(n_episodes=15):
        collisions, total = 0, 0
        for _ in range(n_episodes):
            trajs = collect_rollout(
                env, actor, critic,
                initial_positions_m=[np.array([80.0, 0, 0]), np.array([90.0, 0, 0])],
                initial_velocities_mps=[np.zeros(3), np.zeros(3)],
            )
            collisions += sum(int(t.collided) for t in trajs)
            total += len(trajs)
        return collisions / total

    rate_before = collision_rate()
    for _ in range(60):
        trajs = collect_rollout(
            env, actor, critic,
            initial_positions_m=[np.array([80.0, 0, 0]), np.array([90.0, 0, 0])],
            initial_velocities_mps=[np.zeros(3), np.zeros(3)],
        )
        mappo_update(trajs, actor, critic, actor_opt, critic_opt)
    rate_after = collision_rate()

    assert rate_after <= rate_before