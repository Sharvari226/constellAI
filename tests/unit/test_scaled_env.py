# Tests for the scaling wrappers and the constant-burn reference policy.

import math

import numpy as np
import pytest

from constellai.evaluation.scaled_env import (
    DEFAULT_FUEL_WEIGHT,
    ScaledMultiAgentEnv,
    ScaledSingleAgentEnv,
)
from constellai.evaluation.scenarios import ScenarioConfig
from constellai.evaluation.trials import ConstantBurnPolicy, ZeroPolicy, run_trials
from constellai.models.marl.environment import SingleAgentAvoidanceEnv
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv

CFG = ScenarioConfig()
POSITIONS = [np.array([1500.0, 0.0, 0.0]), np.array([0.0, 2000.0, 0.0])]
VELOCITIES = [np.array([-0.5, 0.0, 0.0]), np.array([0.0, -0.4, 0.0])]


def _raw(n_agents):
    return MultiAgentAvoidanceEnv(
        n_agents=n_agents, mean_motion=CFG.mean_motion, dt=CFG.dt,
        collision_radius_m=CFG.collision_radius_m, max_thrust_mps2=CFG.max_thrust_mps2,
        max_steps=CFG.max_steps,
    )


def _scaled(n_agents):
    return ScaledMultiAgentEnv(_raw(n_agents), n_agents, CFG.max_thrust_mps2, CFG.dt)


def test_observations_are_scaled_to_order_one():
    obs = _scaled(2).reset(POSITIONS, VELOCITIES)
    assert obs[0][0] == pytest.approx(1.5)
    assert obs[1][1] == pytest.approx(2.0)
    assert obs[0][3] == pytest.approx(-0.5)  # velocity untouched
    assert obs[0][6] == pytest.approx(1.0)  # fuel as a fraction of the budget


def test_actions_are_fractions_of_the_thrust_limit():
    scaled, raw = _scaled(2), _raw(2)
    scaled.reset(POSITIONS, VELOCITIES)
    raw.reset(POSITIONS, VELOCITIES)
    action = np.array([0.5, -0.25, 1.0])
    _, scaled_result = scaled.step([action, action])
    _, raw_result = raw.step([action * CFG.max_thrust_mps2, action * CFG.max_thrust_mps2])
    assert scaled_result.infos[0]["separation_m"] == pytest.approx(raw_result.infos[0]["separation_m"])
    assert scaled_result.infos[0]["fuel_remaining"] == pytest.approx(raw_result.infos[0]["fuel_remaining"])


def test_oversized_actions_are_clipped_to_the_limit():
    a, b = _scaled(2), _scaled(2)
    a.reset(POSITIONS, VELOCITIES)
    b.reset(POSITIONS, VELOCITIES)
    _, big = a.step([np.full(3, 5.0), np.full(3, 5.0)])
    _, one = b.step([np.full(3, 1.0), np.full(3, 1.0)])
    assert big.infos[0]["fuel_remaining"] == pytest.approx(one.infos[0]["fuel_remaining"])


def test_full_burn_costs_exactly_the_fuel_weight_per_step():
    scaled, raw = _scaled(2), _raw(2)
    scaled.reset(POSITIONS, VELOCITIES)
    raw.reset(POSITIONS, VELOCITIES)
    full = np.ones(3)
    _, scaled_result = scaled.step([full, full])
    _, raw_result = raw.step([full * CFG.max_thrust_mps2, full * CFG.max_thrust_mps2])
    difference = raw_result.mission_rewards[0] - scaled_result.mission_rewards[0]
    assert difference == pytest.approx(DEFAULT_FUEL_WEIGHT)


def test_zero_thrust_carries_no_fuel_penalty():
    scaled, raw = _scaled(2), _raw(2)
    scaled.reset(POSITIONS, VELOCITIES)
    raw.reset(POSITIONS, VELOCITIES)
    _, scaled_result = scaled.step([np.zeros(3), np.zeros(3)])
    _, raw_result = raw.step([np.zeros(3), np.zeros(3)])
    assert scaled_result.mission_rewards == raw_result.mission_rewards


def test_safety_cost_is_never_modified_by_the_wrapper():
    scaled, raw = _scaled(2), _raw(2)
    close = [np.array([60.0, 0.0, 0.0]), np.array([4000.0, 0.0, 0.0])]
    still = [np.zeros(3), np.zeros(3)]
    scaled.reset(close, still)
    raw.reset(close, still)
    _, scaled_result = scaled.step([np.ones(3), np.ones(3)])
    _, raw_result = raw.step([np.ones(3) * CFG.max_thrust_mps2, np.ones(3) * CFG.max_thrust_mps2])
    assert scaled_result.safety_costs == raw_result.safety_costs


def test_single_agent_wrapper_matches_the_multi_agent_conventions():
    raw = SingleAgentAvoidanceEnv(
        mean_motion=CFG.mean_motion, dt=CFG.dt, collision_radius_m=CFG.collision_radius_m,
        max_thrust_mps2=CFG.max_thrust_mps2, max_steps=CFG.max_steps,
    )
    scaled = ScaledSingleAgentEnv(
        SingleAgentAvoidanceEnv(
            mean_motion=CFG.mean_motion, dt=CFG.dt, collision_radius_m=CFG.collision_radius_m,
            max_thrust_mps2=CFG.max_thrust_mps2, max_steps=CFG.max_steps,
        ),
        CFG.max_thrust_mps2, CFG.dt,
    )
    obs = scaled.reset(POSITIONS[0], VELOCITIES[0])
    raw.reset(POSITIONS[0], VELOCITIES[0])
    assert obs[0] == pytest.approx(1.5)
    assert obs[6] == pytest.approx(1.0)

    full = np.ones(3)
    _, scaled_result = scaled.step(full)
    _, raw_result = raw.step(full * CFG.max_thrust_mps2)
    assert raw_result.mission_reward - scaled_result.mission_reward == pytest.approx(DEFAULT_FUEL_WEIGHT)
    assert scaled_result.safety_cost == raw_result.safety_cost


def test_constant_burn_saturates_every_step_on_a_scaled_env():
    cfg = ScenarioConfig(n_agents=2, max_steps=30)
    env = ScaledMultiAgentEnv(_raw(2), 2, cfg.max_thrust_mps2, cfg.dt)
    trials = run_trials(env, ConstantBurnPolicy(), cfg, n_trials=5)
    assert all(t.saturation_fraction == 1.0 for t in trials)
    assert all(t.mean_fuel_used > 0.0 for t in trials)


def test_zero_policy_still_reports_zero_saturation_on_a_scaled_env():
    cfg = ScenarioConfig(n_agents=2, max_steps=30)
    env = ScaledMultiAgentEnv(_raw(2), 2, cfg.max_thrust_mps2, cfg.dt)
    trials = run_trials(env, ZeroPolicy(), cfg, n_trials=5)
    assert all(t.saturation_fraction == 0.0 for t in trials)