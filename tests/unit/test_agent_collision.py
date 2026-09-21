"""Unit tests for agent-to-agent collision modeling.

test_agents_can_collide_with_each_other is the one that matters most:
it's the first test in the whole project proving satellites can
actually threaten each other, not just a shared external target.
"""

import numpy as np

from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv


def test_agents_can_collide_with_each_other():
    """Two agents placed on top of each other, far from the origin
    threat -- collision must be detected as agent-to-agent, not
    origin-related."""
    env = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=0.0011, collision_radius_m=50.0)
    env.reset(
        initial_positions_m=[np.array([5000.0, 0.0, 0.0]), np.array([5010.0, 0.0, 0.0])],  # 10m apart, far from origin
        initial_velocities_mps=[np.zeros(3), np.zeros(3)],
    )
    _, result = env.step([np.zeros(3), np.zeros(3)])

    assert result.infos[0]["collided_with_agent"] is True
    assert result.infos[0]["collided_with_origin"] is False
    assert result.dones[0] is True


def test_far_apart_agents_do_not_collide_with_each_other():
    env = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=0.0011, collision_radius_m=50.0)
    env.reset(
        initial_positions_m=[np.array([5000.0, 0.0, 0.0]), np.array([9000.0, 0.0, 0.0])],
        initial_velocities_mps=[np.zeros(3), np.zeros(3)],
    )
    _, result = env.step([np.zeros(3), np.zeros(3)])

    assert result.infos[0]["collided_with_agent"] is False


def test_get_threat_positions_excludes_self_and_done_agents():
    env = MultiAgentAvoidanceEnv(n_agents=3, mean_motion=0.0011, collision_radius_m=50.0)
    env.reset(
        initial_positions_m=[np.array([5000.0, 0.0, 0.0]), np.array([6000.0, 0.0, 0.0]), np.array([7000.0, 0.0, 0.0])],
        initial_velocities_mps=[np.zeros(3)] * 3,
    )
    threats = env.get_threat_positions_and_velocities(agent_id=0)
    assert len(threats) == 2  # agents 1 and 2, not itself


def test_done_agent_not_included_as_a_threat_to_others():
    env = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=0.0011, collision_radius_m=500.0)
    env.reset(
        initial_positions_m=[np.array([10.0, 0.0, 0.0]), np.array([5000.0, 0.0, 0.0])],  # agent 0 collides immediately
        initial_velocities_mps=[np.zeros(3), np.zeros(3)],
    )
    env.step([np.zeros(3), np.zeros(3)])  # agent 0 is now done

    threats_seen_by_agent1 = env.get_threat_positions_and_velocities(agent_id=1)
    assert len(threats_seen_by_agent1) == 0  # agent 0 is done, should not appear as a live threat