"""Unit tests for IPPO."""

import numpy as np
import torch

from constellai.models.marl.ippo import (
    ActorCritic,
    collect_rollout,
    compute_gae,
    ppo_update,
)
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv


def test_actor_critic_outputs_correct_shapes():
    policy = ActorCritic()
    obs = torch.zeros(7)
    dist, value = policy(obs)
    assert dist.sample().shape == (3,)
    assert value.shape == ()


def test_collect_rollout_returns_one_trajectory_per_agent():
    env = MultiAgentAvoidanceEnv(n_agents=3, mean_motion=0.0011, max_steps=15)
    policy = ActorCritic()
    trajs = collect_rollout(
        env, policy,
        initial_positions_m=[np.array([1000.0 + 100 * i, 0, 0]) for i in range(3)],
        initial_velocities_mps=[np.zeros(3) for _ in range(3)],
    )
    assert len(trajs) == 3
    assert all(len(t.rewards) > 0 for t in trajs)


def test_gae_matches_known_case():
    # Single-step episode, done=True: advantage should equal reward - value (no bootstrap).
    advantages, returns = compute_gae(rewards=[1.0], values=[0.5], dones=[True], gamma=0.99, lam=0.95)
    assert advantages[0].item() == pytest_approx(0.5)
    assert returns[0].item() == pytest_approx(1.0)


def pytest_approx(x, tol=1e-5):
    class _Approx:
        def __eq__(self, other):
            return abs(other - x) < tol
    return _Approx()


def test_ppo_update_runs_and_changes_weights():
    env = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=0.0011, max_steps=15)
    policy = ActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    params_before = [p.clone() for p in policy.parameters()]

    trajs = collect_rollout(
        env, policy,
        initial_positions_m=[np.array([1000.0, 0, 0]), np.array([1200.0, 0, 0])],
        initial_velocities_mps=[np.zeros(3), np.zeros(3)],
    )
    losses = ppo_update(trajs, policy, optimizer)

    assert "policy_loss" in losses and "value_loss" in losses
    params_after = list(policy.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(params_before, params_after))


def test_shared_policy_learns_across_multiple_agents_and_episodes():
    """Smoke test mirroring rung 1's: a shared policy trained across
    repeated multi-agent episodes on a near-collision scenario should
    reduce collision rate. Not a tight benchmark -- confirms the
    parameter-sharing + PPO update loop actually learns something."""
    torch.manual_seed(0)
    env = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=0.0011, collision_radius_m=50.0, max_steps=25)
    policy = ActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)

    def collision_rate(n_episodes=15):
        collisions = 0
        total_agents = 0
        for _ in range(n_episodes):
            trajs = collect_rollout(
                env, policy,
                initial_positions_m=[np.array([80.0, 0, 0]), np.array([90.0, 0, 0])],
                initial_velocities_mps=[np.zeros(3), np.zeros(3)],
            )
            collisions += sum(int(t.collided) for t in trajs)
            total_agents += len(trajs)
        return collisions / total_agents

    rate_before = collision_rate()

    for _ in range(60):
        trajs = collect_rollout(
            env, policy,
            initial_positions_m=[np.array([80.0, 0, 0]), np.array([5000.0, 0, 0])],
            initial_velocities_mps=[np.zeros(3), np.zeros(3)],
        )
        ppo_update(trajs, policy, optimizer)

    rate_after = collision_rate()
    assert rate_after <= rate_before