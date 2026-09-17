"""Unit tests for M4's single-agent baseline (rung 1)."""

import numpy as np
import torch

from constellai.models.marl.environment import SingleAgentAvoidanceEnv
from constellai.models.marl.single_agent_baseline import (
    GaussianPolicy,
    ValueNet,
    collect_episode,
    compute_discounted_returns,
    reinforce_update,
)


def test_policy_outputs_correct_action_shape():
    policy = GaussianPolicy()
    obs = torch.zeros(7)
    dist = policy(obs)
    action = dist.sample()
    assert action.shape == (3,)


def test_collect_episode_produces_valid_trajectory():
    env = SingleAgentAvoidanceEnv(mean_motion=0.0011, max_steps=20)
    policy = GaussianPolicy()
    traj = collect_episode(
        env, policy,
        initial_position_m=np.array([1000.0, 0, 0]),
        initial_velocity_mps=np.zeros(3),
    )
    assert len(traj.observations) == len(traj.actions) == len(traj.rewards)
    assert len(traj.observations) <= 20
    assert len(traj.observations) > 0


def test_discounted_returns_computed_correctly():
    # Known case: rewards [1, 1, 1], gamma=0.5 -> returns [1.75, 1.5, 1.0]
    returns = compute_discounted_returns([1.0, 1.0, 1.0], gamma=0.5)
    expected = torch.tensor([1.75, 1.5, 1.0])
    torch.testing.assert_close(returns, expected)


def test_reinforce_update_runs_without_error_and_changes_weights():
    env = SingleAgentAvoidanceEnv(mean_motion=0.0011, max_steps=20)
    policy = GaussianPolicy()
    value_net = ValueNet()
    policy_opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=1e-2)

    params_before = [p.clone() for p in policy.parameters()]

    traj = collect_episode(
        env, policy,
        initial_position_m=np.array([1000.0, 0, 0]),
        initial_velocity_mps=np.zeros(3),
    )
    policy_loss, value_loss = reinforce_update(traj, policy, value_net, policy_opt, value_opt)

    assert isinstance(policy_loss, float)
    assert isinstance(value_loss, float)
    params_after = list(policy.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(params_before, params_after))


def test_training_reduces_collision_rate_on_a_trivial_scenario():
    """Sanity check the whole loop actually learns something: starting
    very close to the collision boundary with zero initial velocity, a
    trained policy should collide less often than a freshly initialized
    one, across repeated episodes. Not a tight bound -- this is a smoke
    test for 'does training move the needle at all', not a benchmark."""
    torch.manual_seed(0)
    env = SingleAgentAvoidanceEnv(mean_motion=0.0011, collision_radius_m=50.0, max_steps=30)
    policy = GaussianPolicy()
    value_net = ValueNet()
    policy_opt = torch.optim.Adam(policy.parameters(), lr=5e-3)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=5e-3)

    def collision_rate(n_episodes=20):
        collisions = 0
        for _ in range(n_episodes):
            traj = collect_episode(
                env, policy,
                initial_position_m=np.array([80.0, 0, 0]),
                initial_velocity_mps=np.zeros(3),
            )
            collisions += int(traj.collided)
        return collisions / n_episodes

    rate_before = collision_rate()

    for _ in range(100):
        traj = collect_episode(
            env, policy,
            initial_position_m=np.array([80.0, 0, 0]),
            initial_velocity_mps=np.zeros(3),
        )
        reinforce_update(traj, policy, value_net, policy_opt, value_opt)

    rate_after = collision_rate()

    assert rate_after <= rate_before