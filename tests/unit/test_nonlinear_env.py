# Tests for the independent nonlinear ground-truth simulator.
#
# The two that carry the validation argument are
# test_discrepancy_with_hcw_grows_as_separation_squared and
# test_small_separation_agrees_with_hcw: together they show the simulator and
# HCW agree where HCW is valid and differ at exactly the order theory predicts
# (second order in separation / orbital radius). A frame-conversion or sign bug
# would show up as a first-order error instead.

import numpy as np
import pytest

from constellai.evaluation.nonlinear_env import (
    MU_M3_S2,
    NonlinearAvoidanceEnv,
    eci_to_lvlh,
    lvlh_to_eci,
    propagate_eci_step,
    reference_radius_m,
)
from constellai.evaluation.scaled_env import ScaledMultiAgentEnv
from constellai.evaluation.scenarios import STREAM_EVAL, ScenarioConfig, make_rng, sample_scenario
from constellai.evaluation.trials import ConstantBurnPolicy, ZeroPolicy, run_trials
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv
from constellai.orbital_mechanics.relative_motion import RelativeState, propagate_hcw_step

CFG = ScenarioConfig()
N = CFG.mean_motion
A = reference_radius_m(N)
ORBIT_STEPS = int(round(2 * np.pi / N / CFG.dt))


def _nonlinear(n_agents, **overrides):
    kwargs = dict(
        n_agents=n_agents, mean_motion=CFG.mean_motion, dt=CFG.dt,
        collision_radius_m=CFG.collision_radius_m, max_thrust_mps2=CFG.max_thrust_mps2,
        max_steps=CFG.max_steps,
    )
    kwargs.update(overrides)
    return NonlinearAvoidanceEnv(**kwargs)


def _hcw(n_agents):
    return MultiAgentAvoidanceEnv(
        n_agents=n_agents, mean_motion=CFG.mean_motion, dt=CFG.dt,
        collision_radius_m=CFG.collision_radius_m, max_thrust_mps2=CFG.max_thrust_mps2,
        max_steps=CFG.max_steps,
    )


def _bounded_orbit(offset_m):
    # HCW initial condition with no secular drift: along-track velocity = -2 n x
    return np.array([offset_m, 0.0, 0.0]), np.array([0.0, -2.0 * N * offset_m, 0.0])


def _max_error_vs_hcw_over_one_orbit(offset_m):
    rho, rho_dot = _bounded_orbit(offset_m)
    r, v = lvlh_to_eci(rho, rho_dot, A, N, 0.0)
    state = np.concatenate([r, v])
    hcw = RelativeState(position=rho.copy(), velocity=rho_dot.copy())
    t, worst = 0.0, 0.0
    for _ in range(ORBIT_STEPS):
        state = propagate_eci_step(state, t, CFG.dt, np.zeros(3), N, 2)
        hcw = propagate_hcw_step(hcw, np.zeros(3), N, CFG.dt)
        t += CFG.dt
        position, _ = eci_to_lvlh(state[:3], state[3:], A, N, t)
        worst = max(worst, float(np.linalg.norm(position - hcw.position)))
    return worst


def test_reference_radius_matches_the_500km_shell():
    assert A / 1000.0 == pytest.approx(6878.135, abs=1e-3)


def test_lvlh_eci_roundtrip_is_exact():
    rng = np.random.default_rng(0)
    for t in (0.0, 300.0, 2500.0):
        rho, rho_dot = rng.normal(size=3) * 1000.0, rng.normal(size=3)
        r, v = lvlh_to_eci(rho, rho_dot, A, N, t)
        rho2, rho_dot2 = eci_to_lvlh(r, v, A, N, t)
        np.testing.assert_allclose(rho, rho2, atol=1e-6)
        np.testing.assert_allclose(rho_dot, rho_dot2, atol=1e-6)


def test_two_body_integration_conserves_energy_and_angular_momentum():
    rho, rho_dot = _bounded_orbit(5000.0)
    r, v = lvlh_to_eci(rho, rho_dot, A, N, 0.0)
    state = np.concatenate([r, v])
    energy0 = 0.5 * v @ v - MU_M3_S2 / np.linalg.norm(r)
    h0 = np.cross(r, v)
    t = 0.0
    for _ in range(ORBIT_STEPS):
        state = propagate_eci_step(state, t, CFG.dt, np.zeros(3), N, 2)
        t += CFG.dt
    energy1 = 0.5 * state[3:] @ state[3:] - MU_M3_S2 / np.linalg.norm(state[:3])
    h1 = np.cross(state[:3], state[3:])
    assert abs((energy1 - energy0) / energy0) < 1e-10
    assert np.linalg.norm(h1 - h0) / np.linalg.norm(h0) < 1e-10


def test_co_orbiting_agent_stays_at_the_origin():
    env = _nonlinear(1)
    env.reset([np.zeros(3)], [np.zeros(3)])
    for _ in range(100):
        obs, _ = env.step([np.zeros(3)])
    assert np.linalg.norm(obs[0][:3]) < 0.01


def test_small_separation_agrees_with_hcw():
    assert _max_error_vs_hcw_over_one_orbit(100.0) < 0.05


def test_discrepancy_with_hcw_grows_as_separation_squared():
    e100 = _max_error_vs_hcw_over_one_orbit(100.0)
    e200 = _max_error_vs_hcw_over_one_orbit(200.0)
    e400 = _max_error_vs_hcw_over_one_orbit(400.0)
    assert 3.0 < e200 / e100 < 5.0
    assert 3.0 < e400 / e200 < 5.0


def test_thrust_is_applied_in_the_lvlh_frame_like_the_hcw_environment():
    positions = [np.array([800.0, 300.0, -100.0]), np.array([-600.0, 900.0, 200.0])]
    velocities = [np.array([-0.4, 0.2, 0.0]), np.array([0.3, -0.5, 0.1])]
    thrust = [np.array([0.005, -0.003, 0.002]), np.array([-0.004, 0.006, -0.001])]
    nonlinear, hcw = _nonlinear(2), _hcw(2)
    nonlinear.reset(positions, velocities)
    hcw.reset(positions, velocities)
    for _ in range(30):
        obs_n, _ = nonlinear.step(thrust)
        obs_h, _ = hcw.step(thrust)
    for a, b in zip(obs_n, obs_h):
        assert np.linalg.norm(a[:3] - b[:3]) < 0.5
        assert a[6] == pytest.approx(b[6])  # same fuel accounting


def test_agent_to_agent_collision_is_detected():
    env = _nonlinear(2)
    env.reset([np.array([5000.0, 0, 0]), np.array([5010.0, 0, 0])], [np.zeros(3), np.zeros(3)])
    _, result = env.step([np.zeros(3), np.zeros(3)])
    assert result.infos[0]["collided_with_agent"] is True
    assert result.infos[0]["collided_with_origin"] is False
    assert result.dones[0] is True


def test_fuel_accounting_and_thrust_clipping():
    env = _nonlinear(1)
    env.reset([np.array([3000.0, 0, 0])], [np.zeros(3)])
    _, result = env.step([np.full(3, 5.0)])  # far above the limit
    expected = CFG.max_thrust_mps2 * np.sqrt(3.0) * CFG.dt
    assert result.infos[0]["fuel_remaining"] == pytest.approx(env.fuel_budget - expected)


def test_threats_exclude_self_and_done_agents():
    env = _nonlinear(3)
    env.reset([np.array([5000.0, 0, 0]), np.array([6000.0, 0, 0]), np.array([7000.0, 0, 0])], [np.zeros(3)] * 3)
    assert len(env.get_threat_positions_and_velocities(0)) == 2

    env = _nonlinear(2, collision_radius_m=500.0)
    env.reset([np.array([10.0, 0, 0]), np.array([5000.0, 0, 0])], [np.zeros(3), np.zeros(3)])
    env.step([np.zeros(3), np.zeros(3)])  # agent 0 collides and is done
    assert env.get_threat_positions_and_velocities(1) == []


def test_reset_rejects_a_wrong_agent_count():
    with pytest.raises(ValueError):
        _nonlinear(2).reset([np.zeros(3)], [np.zeros(3)])


def test_scenarios_keep_their_meaning_in_the_ground_truth_simulator():
    # scenarios are BUILT with HCW; zero-thrust outcomes must match in the nonlinear truth
    mismatches = 0
    for index in range(100):
        positions, velocities = sample_scenario(make_rng(STREAM_EVAL, 0, index), CFG)
        outcomes = []
        for env in (_hcw(CFG.n_agents), _nonlinear(CFG.n_agents)):
            env.reset(positions, velocities)
            collided, done = False, False
            while not done:
                _, result = env.step([np.zeros(3)] * CFG.n_agents)
                collided = collided or any(i.get("collided", False) for i in result.infos)
                done = all(result.dones)
            outcomes.append(collided)
        mismatches += outcomes[0] != outcomes[1]
    assert mismatches <= 3


def test_runs_under_the_scaling_wrapper_and_trial_runner():
    cfg = ScenarioConfig(n_agents=2, max_steps=30)
    env = ScaledMultiAgentEnv(_nonlinear(2, max_steps=30), 2, cfg.max_thrust_mps2, cfg.dt)
    burn = run_trials(env, ConstantBurnPolicy(), cfg, n_trials=3)
    zero = run_trials(env, ZeroPolicy(), cfg, n_trials=3)
    assert all(t.saturation_fraction == 1.0 for t in burn)
    assert all(t.mean_fuel_used == 0.0 for t in zero)


def test_m5_filter_wrapper_accepts_the_simulator():
    # compatibility check for the Phase 4 composition; skipped where scipy / the safety package is missing
    filtered_module = pytest.importorskip("constellai.safety.filtered_environment")
    env = filtered_module.FilteredMultiAgentEnv(_nonlinear(2))
    env.reset([np.array([60.0, 0, 0]), np.array([5000.0, 0, 0])], [np.array([-5.0, 0, 0]), np.zeros(3)])
    env.step([np.zeros(3), np.zeros(3)])
    assert env.stats.total_actions == 2