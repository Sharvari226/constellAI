import numpy as np
import pytest
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv

N = 0.0011  # rad/s, LEO-ish; match your other env tests if different
Z = np.zeros(3)

def make():
    return MultiAgentAvoidanceEnv(n_agents=2, mean_motion=N)

def run(env, pos, thrusts):
    env.reset(pos, [Z.copy(), Z.copy()])
    return env.step(thrusts)[1]

def test_hold_distance_is_free():
    # both agents ~1000 m from origin, far apart: no cost, ~zero reward
    env = make()
    assert 5 * env.collision_radius_m < 1000.0
    res = run(env, [np.array([1000.0, 0, 0]), np.array([0, 1000.0, 0])], [Z, Z])
    assert max(res.safety_costs) == 0.0
    assert all(abs(r) < 1e-9 for r in res.mission_rewards)

def test_collision_costs_one():
    env = make()
    R = env.collision_radius_m
    a = np.array([1000.0, 0, 0])
    res = run(env, [a, a + np.array([0.5 * R, 0, 0])], [Z, Z])
    assert res.safety_costs[0] == 1.0

def test_warning_zone_is_shaped_not_binary():
    env = make()
    R = env.collision_radius_m
    a = np.array([1000.0, 0, 0])
    res = run(env, [a, a + np.array([3.0 * R, 0, 0])], [Z, Z])
    assert 0.0 < res.safety_costs[0] < 1.0

def test_secondary_cost_requires_burn():
    env = make()
    R = env.collision_radius_m
    a = np.array([1000.0, 0, 0])
    pos = [a, a + np.array([3.0 * R, 0, 0])]
    idle = run(env, pos, [Z, Z])
    assert idle.infos[0]["cost_sec"] == 0.0
    burn = run(make(), pos, [np.array([0.5 * env.max_thrust_mps2, 0, 0]), Z])
    assert burn.infos[0]["cost_sec"] > 0.0

def test_fuel_not_double_counted():
    # the fuel penalty lives in ScaledMultiAgentEnv; the raw env must not add its own
    env = make()
    env.reset([np.array([1000.0, 0, 0]), np.array([0, 1000.0, 0])], [Z.copy(), Z.copy()])
    assert env._obj_cfg.w_fuel == 0.0
    res = env.step([np.array([env.max_thrust_mps2, 0, 0]), Z])[1]
    assert res.infos[0]["fuel_remaining"] < env.fuel_budget   # burn happened

def test_pair_collision_is_symmetric():
    env = make()
    R = env.collision_radius_m
    a = np.array([1000.0, 0, 0])
    res = run(env, [a, a + np.array([0.5 * R, 0, 0])], [Z, Z])
    assert all(i["collided_with_agent"] for i in res.infos)
    assert all(res.dones)
    assert res.safety_costs[0] == res.safety_costs[1] == 1.0

def test_pair_cost_is_symmetric_under_burn():
    # agent 1 burns toward agent 0; both must score the same post-move geometry
    env = make()
    R = env.collision_radius_m
    a = np.array([1000.0, 0, 0])
    b = a + np.array([3.0 * R, 0, 0])
    res = run(env, [a, b], [Z, np.array([-env.max_thrust_mps2, 0, 0])])
    assert abs(res.safety_costs[0] - res.safety_costs[1]) < 1e-9