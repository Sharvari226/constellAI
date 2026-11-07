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


def test_zero_thrust_earns_zero_mission_reward_from_any_start():
    from constellai.evaluation.scenarios import STREAM_FC_VAL, make_rng, sample_scenario, warning_config
    cfg = warning_config("short")
    for idx in range(5):
        pos, vel = sample_scenario(make_rng(STREAM_FC_VAL, 0, idx), cfg)
        env = MultiAgentAvoidanceEnv(n_agents=cfg.n_agents, mean_motion=cfg.mean_motion)
        env.reset(pos, vel)
        for _ in range(20):
            _, res = env.step([Z for _ in range(cfg.n_agents)])
            assert all(abs(r) < 1e-12 for r in res.mission_rewards)
            if all(res.dones):
                break


def test_nominal_path_is_independent_zero_thrust_propagation():
    from constellai.orbital_mechanics.relative_motion import RelativeState, propagate_hcw_step
    env = make()
    env.reset([np.array([1000.0, 0, 0]), np.array([0, 1000.0, 0])],
              [np.array([0.0, 0.3, 0.0]), np.array([0.2, 0.0, 0.0])])
    nom = RelativeState(position=np.array([1000.0, 0, 0]), velocity=np.array([0.0, 0.3, 0.0]))
    act = RelativeState(position=nom.position.copy(), velocity=nom.velocity.copy())
    u = np.array([0.004, -0.002, 0.001])
    for _ in range(12):
        env.step([u, Z])
        nom = propagate_hcw_step(nom, Z, N, env.dt)
        act = propagate_hcw_step(act, u, N, env.dt)
    np.testing.assert_allclose(env._nominal[0].position, nom.position, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(env._states[0].position, act.position, rtol=1e-12, atol=1e-9)
    assert np.linalg.norm(env._states[0].position - env._nominal[0].position) > 1.0


def test_primary_shell_mode_reproduces_old_term():
    pos = [np.array([300.0, 0, 0]), np.array([0, 1000.0, 0])]
    env = make()
    env.mission_mode = "primary_shell"
    env.reset(pos, [Z.copy(), Z.copy()])
    _, res = env.step([Z, Z])
    sep = res.infos[0]["separation_to_origin_m"]
    expected = -min(max(0.0, abs(sep - 1000.0) - 50.0) / 1000.0, 3.0)
    assert abs(res.mission_rewards[0] - expected) < 1e-12 and expected < -0.5
    env2 = make()
    env2.reset(pos, [Z.copy(), Z.copy()])
    _, res2 = env2.step([Z, Z])
    assert abs(res2.mission_rewards[0]) < 1e-12
