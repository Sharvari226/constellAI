"""Unit tests for the single-agent avoidance environment."""

import numpy as np
import pytest

from constellai.models.marl.environment import SingleAgentAvoidanceEnv


def test_reset_returns_correct_observation_shape():
    env = SingleAgentAvoidanceEnv(mean_motion=0.0011)
    obs = env.reset(initial_position_m=np.array([1000.0, 0, 0]), initial_velocity_mps=np.zeros(3))
    assert obs.shape == (7,)
    assert obs[-1] == pytest.approx(env.fuel_budget)


def test_step_before_reset_raises():
    env = SingleAgentAvoidanceEnv(mean_motion=0.0011)
    with pytest.raises(RuntimeError):
        env.step(np.zeros(3))


def test_zero_thrust_consumes_no_fuel():
    env = SingleAgentAvoidanceEnv(mean_motion=0.0011)
    env.reset(initial_position_m=np.array([1000.0, 0, 0]), initial_velocity_mps=np.zeros(3))
    obs, result = env.step(np.zeros(3))
    assert result.info["fuel_remaining"] == pytest.approx(env.fuel_budget)


def test_nonzero_thrust_consumes_fuel():
    env = SingleAgentAvoidanceEnv(mean_motion=0.0011)
    env.reset(initial_position_m=np.array([1000.0, 0, 0]), initial_velocity_mps=np.zeros(3))
    _, result = env.step(np.array([0.005, 0, 0]))
    assert result.info["fuel_remaining"] < env.fuel_budget


def test_thrust_is_clipped_to_max():
    env = SingleAgentAvoidanceEnv(mean_motion=0.0011, max_thrust_mps2=0.01)
    env.reset(initial_position_m=np.array([1000.0, 0, 0]), initial_velocity_mps=np.zeros(3))
    _, result_huge = env.step(np.array([1000.0, 0, 0]))
    env.reset(initial_position_m=np.array([1000.0, 0, 0]), initial_velocity_mps=np.zeros(3))
    _, result_max = env.step(np.array([0.01, 0, 0]))
    # A wildly oversized action should consume exactly the same fuel as
    # the max allowed action, not more -- proves clipping actually happened.
    assert result_huge.info["fuel_remaining"] == pytest.approx(result_max.info["fuel_remaining"])


def test_collision_sets_safety_cost_and_ends_episode():
    env = SingleAgentAvoidanceEnv(mean_motion=0.0011, collision_radius_m=50.0)
    env.reset(initial_position_m=np.array([10.0, 0, 0]), initial_velocity_mps=np.zeros(3))
    _, result = env.step(np.zeros(3))
    assert result.safety_cost == 1.0
    assert result.done is True
    assert result.info["collided"] is True


def test_no_collision_gives_zero_safety_cost():
    env = SingleAgentAvoidanceEnv(mean_motion=0.0011, collision_radius_m=50.0)
    env.reset(initial_position_m=np.array([1000.0, 0, 0]), initial_velocity_mps=np.zeros(3))
    _, result = env.step(np.zeros(3))
    assert result.safety_cost == 0.0
    assert result.done is False


def test_episode_ends_when_fuel_exhausted():
    env = SingleAgentAvoidanceEnv(mean_motion=0.0011, fuel_budget=0.001, max_thrust_mps2=0.01)
    env.reset(initial_position_m=np.array([1000.0, 0, 0]), initial_velocity_mps=np.zeros(3))
    _, result = env.step(np.array([0.01, 0, 0]))
    assert result.info["fuel_remaining"] == 0.0
    assert result.done is True


def test_episode_ends_at_max_steps_if_nothing_else_happens():
    env = SingleAgentAvoidanceEnv(mean_motion=0.0011, max_steps=3, collision_radius_m=1.0)
    env.reset(initial_position_m=np.array([100000.0, 0, 0]), initial_velocity_mps=np.zeros(3))
    for i in range(3):
        _, result = env.step(np.zeros(3))
    assert result.done is True
    assert result.info["step_count"] == 3