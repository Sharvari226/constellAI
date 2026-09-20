"""Unit tests for the multi-agent avoidance environment, focused
specifically on the independence contract -- that's the property this
environment's whole reason for existing depends on, more than any
single physics detail (already covered by test_relative_motion.py)."""

import numpy as np
import pytest

from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv


def test_reset_requires_matching_agent_count():
    env = MultiAgentAvoidanceEnv(n_agents=3, mean_motion=0.0011)
    with pytest.raises(ValueError):
        env.reset(
            initial_positions_m=[np.array([1000.0, 0, 0])] * 2,  # only 2, not 3
            initial_velocities_mps=[np.zeros(3)] * 2,
        )


def test_reset_returns_one_observation_per_agent():
    env = MultiAgentAvoidanceEnv(n_agents=3, mean_motion=0.0011)
    obs = env.reset(
        initial_positions_m=[np.array([1000.0, 0, 0]), np.array([2000.0, 0, 0]), np.array([3000.0, 0, 0])],
        initial_velocities_mps=[np.zeros(3)] * 3,
    )
    assert len(obs) == 3
    assert all(o.shape == (7,) for o in obs)


def test_agents_do_not_see_each_others_state():
    """The core independence property: agent 0's observation must be
    identical whether agent 1 starts at 2000m or 20000m away -- if it
    isn't, information is leaking between agents, breaking the whole
    premise of this being an IPPO-appropriate environment."""
    env_a = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=0.0011)
    obs_a = env_a.reset(
        initial_positions_m=[np.array([1000.0, 0, 0]), np.array([2000.0, 0, 0])],
        initial_velocities_mps=[np.zeros(3)] * 2,
    )

    env_b = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=0.0011)
    obs_b = env_b.reset(
        initial_positions_m=[np.array([1000.0, 0, 0]), np.array([20000.0, 0, 0])],
        initial_velocities_mps=[np.zeros(3)] * 2,
    )

    np.testing.assert_allclose(obs_a[0], obs_b[0])  # agent 0's obs unaffected by agent 1's position


def test_step_requires_one_action_per_agent():
    env = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=0.0011)
    env.reset(initial_positions_m=[np.array([1000.0, 0, 0])] * 2, initial_velocities_mps=[np.zeros(3)] * 2)
    with pytest.raises(ValueError):
        env.step([np.zeros(3)])  # only 1 action for 2 agents


def test_each_agent_gets_independent_rewards_and_costs():
    env = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=0.0011, collision_radius_m=50.0)
    env.reset(
        initial_positions_m=[np.array([10.0, 0, 0]), np.array([5000.0, 0, 0])],  # agent 0 near collision, agent 1 far
        initial_velocities_mps=[np.zeros(3)] * 2,
    )
    _, result = env.step([np.zeros(3), np.zeros(3)])

    assert result.safety_costs[0] == 1.0  # agent 0 collided
    assert result.safety_costs[1] == 0.0  # agent 1 did not
    assert result.dones[0] is True
    assert result.dones[1] is False


def test_done_agent_is_not_restepped():
    env = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=0.0011, collision_radius_m=50.0)
    env.reset(
        initial_positions_m=[np.array([10.0, 0, 0]), np.array([5000.0, 0, 0])],
        initial_velocities_mps=[np.zeros(3)] * 2,
    )
    _, result1 = env.step([np.zeros(3), np.zeros(3)])
    assert result1.dones[0] is True

    _, result2 = env.step([np.array([0.01, 0, 0]), np.zeros(3)])  # try to move agent 0 anyway
    assert result2.infos[0].get("already_done") is True
    assert result2.mission_rewards[0] == 0.0
    