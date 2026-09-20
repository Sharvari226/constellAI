"""Unit tests for constrained/Lagrangian MAPPO (rung 5, the proposed method).

test_lambda_increases_when_constraint_violated and
test_lambda_decreases_when_constraint_comfortably_satisfied are the
two that matter most -- they're the concrete proof this rung actually
does automatic constraint adaptation, not just "MAPPO with an extra
critic that does nothing."
"""

import numpy as np
import torch

from constellai.models.marl.constrained_mappo import (
    collect_rollout,
    constrained_update,
)
from constellai.models.marl.mappo import Actor, CentralizedCritic
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv


def _make_networks(n_agents=2):
    actor = Actor()
    reward_critic = CentralizedCritic(n_agents=n_agents)
    cost_critic = CentralizedCritic(n_agents=n_agents)
    return actor, reward_critic, cost_critic


def test_rollout_keeps_reward_and_cost_separate():
    env = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=0.0011, max_steps=10)
    actor, reward_critic, cost_critic = _make_networks()
    trajs = collect_rollout(
        env, actor, reward_critic, cost_critic,
        initial_positions_m=[np.array([1000.0, 0, 0]), np.array([1200.0, 0, 0])],
        initial_velocities_mps=[np.zeros(3), np.zeros(3)],
    )
    # mission_rewards and safety_costs must be tracked as genuinely
    # separate lists, never combined -- this is the entire point of
    # this rung existing.
    assert len(trajs[0].mission_rewards) == len(trajs[0].safety_costs)
    assert trajs[0].mission_rewards != trajs[0].safety_costs  # sanity: not accidentally aliased


def test_constrained_update_returns_expected_keys():
    env = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=0.0011, max_steps=15)
    actor, reward_critic, cost_critic = _make_networks()
    trajs = collect_rollout(
        env, actor, reward_critic, cost_critic,
        initial_positions_m=[np.array([1000.0, 0, 0]), np.array([1200.0, 0, 0])],
        initial_velocities_mps=[np.zeros(3), np.zeros(3)],
    )
    result = constrained_update(
        trajs, actor, reward_critic, cost_critic, lagrange_lambda=1.0,
        actor_optimizer=torch.optim.Adam(actor.parameters()),
        reward_critic_optimizer=torch.optim.Adam(reward_critic.parameters()),
        cost_critic_optimizer=torch.optim.Adam(cost_critic.parameters()),
    )
    assert set(result.keys()) == {
        "policy_loss", "reward_value_loss", "cost_value_loss", "mean_cost_return", "lambda"
    }


def test_lambda_increases_when_constraint_violated():
    """Force a high measured cost return (well above the threshold) and
    confirm lambda grows -- the actual dual-ascent mechanism, not just
    a returned number that happens not to move."""
    env = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=0.0011, collision_radius_m=500.0, max_steps=15)
    actor, reward_critic, cost_critic = _make_networks()
    # Start VERY close, at a radius guaranteed to trigger collisions
    # given the deliberately large collision_radius_m above -- forces a
    # high realized safety_cost, hence a high cost return.
    trajs = collect_rollout(
        env, actor, reward_critic, cost_critic,
        initial_positions_m=[np.array([10.0, 0, 0]), np.array([15.0, 0, 0])],
        initial_velocities_mps=[np.zeros(3), np.zeros(3)],
    )
    result = constrained_update(
        trajs, actor, reward_critic, cost_critic, lagrange_lambda=0.0,
        actor_optimizer=torch.optim.Adam(actor.parameters()),
        reward_critic_optimizer=torch.optim.Adam(reward_critic.parameters()),
        cost_critic_optimizer=torch.optim.Adam(cost_critic.parameters()),
        cost_threshold_d=0.01,  # very low threshold, easy to violate here
    )
    assert result["lambda"] > 0.0


def test_lambda_decreases_when_constraint_comfortably_satisfied():
    """The reverse: start far away (near-zero collision risk), a
    generous cost threshold -> lambda should stay at or near 0, not
    grow -- confirms the mechanism responds to ACTUAL measured safety,
    not just always ratcheting upward."""
    env = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=0.0011, collision_radius_m=1.0, max_steps=15)
    actor, reward_critic, cost_critic = _make_networks()
    trajs = collect_rollout(
        env, actor, reward_critic, cost_critic,
        initial_positions_m=[np.array([100000.0, 0, 0]), np.array([120000.0, 0, 0])],
        initial_velocities_mps=[np.zeros(3), np.zeros(3)],
    )
    result = constrained_update(
        trajs, actor, reward_critic, cost_critic, lagrange_lambda=0.5,
        actor_optimizer=torch.optim.Adam(actor.parameters()),
        reward_critic_optimizer=torch.optim.Adam(reward_critic.parameters()),
        cost_critic_optimizer=torch.optim.Adam(cost_critic.parameters()),
        cost_threshold_d=0.5,  # generous threshold, easily satisfied here
    )
    assert result["lambda"] < 0.5  # should shrink from its starting value


def test_full_constrained_training_loop_learns_and_tracks_cost():
    """Smoke test mirroring every prior rung's, but ALSO checks that
    mean_cost_return trends downward across training, not just that
    collision rate does -- the cost-critic-specific claim this rung
    adds on top of what MAPPO already demonstrated."""
    torch.manual_seed(0)
    env = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=0.0011, collision_radius_m=50.0, max_steps=25)
    actor, reward_critic, cost_critic = _make_networks()
    actor_opt = torch.optim.Adam(actor.parameters(), lr=5e-3)
    reward_critic_opt = torch.optim.Adam(reward_critic.parameters(), lr=5e-3)
    cost_critic_opt = torch.optim.Adam(cost_critic.parameters(), lr=5e-3)

    lam = 1.0
    cost_returns_over_time = []
    for _ in range(60):
        trajs = collect_rollout(
            env, actor, reward_critic, cost_critic,
            initial_positions_m=[np.array([80.0, 0, 0]), np.array([90.0, 0, 0])],
            initial_velocities_mps=[np.zeros(3), np.zeros(3)],
        )
        result = constrained_update(
            trajs, actor, reward_critic, cost_critic, lagrange_lambda=lam,
            actor_optimizer=actor_opt,
            reward_critic_optimizer=reward_critic_opt,
            cost_critic_optimizer=cost_critic_opt,
        )
        lam = result["lambda"]
        cost_returns_over_time.append(result["mean_cost_return"])

    first_half_avg = sum(cost_returns_over_time[:20]) / 20
    second_half_avg = sum(cost_returns_over_time[-20:]) / 20
    assert second_half_avg <= first_half_avg