import numpy as np
import pytest

from constellai.evaluation.scaled_env import ScaledMultiAgentEnv
from constellai.evaluation.scenarios import (
    STREAM_FC_VAL, make_rng, reference_mean_motion, sample_scenario, warning_config,
)
from constellai.forecasting.events import (
    AnalyticForecaster, FirstEntryEngine, cumulative_labels, horizon_steps,
    interval_index, pair_states,
)
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv
from constellai.orbital_mechanics.relative_motion import RelativeState, propagate_hcw_step
from constellai.tracking.tracker import TrackedMultiAgentEnv

N = reference_mean_motion()
CFG = warning_config("short")


def scenario(index):
    pos, vel = sample_scenario(make_rng(STREAM_FC_VAL, 0, index), CFG)
    return pos, vel, [np.concatenate([p, v]) for p, v in zip(pos, vel)]


def brute_first_entry(x0):
    state = RelativeState.from_vector(np.asarray(x0, dtype=float))
    for s in range(1, CFG.max_steps + 1):
        state = propagate_hcw_step(state, np.zeros(3), N, CFG.dt)
        if np.linalg.norm(state.position) < CFG.collision_radius_m:
            return s
    return 0


def forecaster(seed=0, n=2000):
    return AnalyticForecaster(N, CFG.dt, CFG.collision_radius_m, n_samples=n, seed=seed)


def test_first_entry_matches_propagator():
    eng = FirstEntryEngine(N, CFG.dt, CFG.collision_radius_m, CFG.max_steps)
    xs = [x for idx in range(40) for _, _, x in pair_states(scenario(idx)[2])]
    first = eng.first_entry(np.array(xs))
    ref = np.array([brute_first_entry(x) for x in xs])
    np.testing.assert_array_equal(first, ref)
    assert (ref > 0).any() and (ref == 0).any()


def test_first_entry_matches_env_first_collision():
    raw = MultiAgentAvoidanceEnv(n_agents=CFG.n_agents, mean_motion=CFG.mean_motion, dt=CFG.dt,
                                 collision_radius_m=CFG.collision_radius_m,
                                 max_thrust_mps2=CFG.max_thrust_mps2, max_steps=CFG.max_steps)
    eng = FirstEntryEngine(N, CFG.dt, CFG.collision_radius_m, CFG.max_steps)
    n_events = 0
    for idx in range(100):
        pos, vel, truth = scenario(idx)
        firsts = eng.first_entry(np.array([x for _, _, x in pair_states(truth)]))
        expected = int(firsts[firsts > 0].min()) if (firsts > 0).any() else 0
        raw.reset(pos, vel)
        actual, step_no, done = 0, 0, False
        while not done:
            _, res = raw.step([np.zeros(3) for _ in range(CFG.n_agents)])
            step_no += 1
            if actual == 0 and any(info.get("collided", False) for info in res.infos):
                actual = step_no
            done = all(res.dones)
        assert actual == expected, (idx, actual, expected)
        n_events += expected > 0
    assert n_events > 10


def test_cumulative_labels_and_intervals():
    steps = horizon_steps(10.0)
    assert steps == [10, 20, 40, 60]
    first = np.array([0, 5, 10, 11, 20, 21, 60, 61])
    np.testing.assert_array_equal(interval_index(first, steps), [0, 1, 1, 2, 2, 3, 4, 0])
    cum = cumulative_labels(first, steps)
    assert (np.diff(cum, axis=1) >= 0).all()
    np.testing.assert_array_equal(cum[3], [0, 1, 1, 1])


def test_horizon_steps_validation():
    with pytest.raises(ValueError):
        horizon_steps(10.0, (105.0, 200.0))
    with pytest.raises(ValueError):
        horizon_steps(10.0, (200.0, 100.0))


def test_forecast_monotone_bounded_and_dominates_instantaneous():
    f = forecaster()
    rng = np.random.default_rng(0)
    for idx in range(20):
        for _, _, x0 in pair_states(scenario(idx)[2]):
            a = rng.uniform(1.0, 60.0)
            cov = np.diag([a * a] * 3 + [0.25] * 3)
            mean = x0 + np.concatenate([rng.normal(0, a, 3), rng.normal(0, 0.3, 3)])
            p, q = f(mean, cov), f.instantaneous(mean, cov)
            assert (np.diff(p) >= -1e-12).all() and ((p >= 0) & (p <= 1)).all()
            assert (q <= p + 1e-12).all()


def test_zero_uncertainty_reproduces_deterministic_labels():
    f = forecaster(n=50)
    n_pos = 0
    for idx in range(30):
        for _, _, x0 in pair_states(scenario(idx)[2]):
            truth = cumulative_labels(f.engine.first_entry(x0), f.steps)[0]
            np.testing.assert_array_equal(f(x0, 1e-10 * np.eye(6)), truth)
            n_pos += truth[-1]
    assert n_pos > 0


def test_singular_covariance_does_not_crash():
    f = forecaster(n=100)
    x0 = pair_states(scenario(0)[2])[0][2]
    assert np.isfinite(f(x0, np.zeros((6, 6)))).all()


def test_monte_carlo_seed_agreement():
    a, b = forecaster(seed=0), forecaster(seed=1)
    cov = np.diag([30.0 ** 2] * 3 + [0.3 ** 2] * 3)
    for idx in range(10):
        for _, _, x0 in pair_states(scenario(idx)[2]):
            assert np.abs(a(x0, cov) - b(x0, cov)).max() < 0.08


def test_tracker_hook_integration():
    raw = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=N)
    scaled = ScaledMultiAgentEnv(raw, 2, raw.max_thrust_mps2, raw.dt)
    fc = forecaster(n=200)
    env = TrackedMultiAgentEnv(scaled, sigma_p=25.0, process_accel_std=0.0, forecaster=fc,
                               neighbour_thrust_known=True)
    obs = env.reset([np.array([1000.0, 0, 0]), np.array([0, 1000.0, 0])], [np.zeros(3), np.zeros(3)])
    mean, cov, _ = env.pair_track(0)
    np.testing.assert_allclose(obs[0][33:37], fc(mean, cov))
    assert obs[0].shape == (41,) and ((obs[0][33:] >= 0) & (obs[0][33:] <= 1)).all()


def test_oracle_forecast_is_exact_and_changes_nothing_else():
    def build(oracle):
        raw = MultiAgentAvoidanceEnv(n_agents=3, mean_motion=N)
        scaled = ScaledMultiAgentEnv(raw, 3, raw.max_thrust_mps2, raw.dt)
        return TrackedMultiAgentEnv(scaled, sigma_p=25.0, process_accel_std=0.0,
                                    neighbour_thrust_known=True, oracle=oracle)
    oracle = forecaster(n=10)
    a, b = build(None), build(oracle)
    assert a.obs_dim == 33 and b.obs_dim == 41
    for idx in range(6):
        pos, vel, truth = scenario(idx)
        oa = a.reset(pos, vel, episode_index=idx)
        ob = b.reset(pos, vel, episode_index=idx)
        for i in range(3):
            np.testing.assert_array_equal(oa[i], ob[i][:33])
            first = oracle.engine.first_entry(-truth[i])
            np.testing.assert_array_equal(ob[i][33:37], cumulative_labels(first, oracle.steps)[0])
            j = b.slot1_index(i)
            assert j is not None
            first_j = oracle.engine.first_entry(truth[j] - truth[i])
            np.testing.assert_array_equal(ob[i][37:41], cumulative_labels(first_j, oracle.steps)[0])


def test_forecaster_and_oracle_are_mutually_exclusive():
    raw = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=N)
    scaled = ScaledMultiAgentEnv(raw, 2, raw.max_thrust_mps2, raw.dt)
    with pytest.raises(AssertionError):
        TrackedMultiAgentEnv(scaled, sigma_p=25.0, process_accel_std=0.0,
                             forecaster=forecaster(n=10), oracle=forecaster(n=10))


def test_gate_skips_only_negligible_risk():
    gated = AnalyticForecaster(N, CFG.dt, CFG.collision_radius_m, n_samples=2000, seed=0, gate_sigmas=6.0)
    plain = forecaster()
    rng = np.random.default_rng(1)
    fired = passed = 0
    for idx in range(25):
        for _, _, x0 in pair_states(scenario(idx)[2]):
            a = rng.uniform(0.5, 6.0)
            cov = np.diag([a * a] * 3 + [0.25] * 3)
            mean = x0 + np.concatenate([rng.normal(0, a, 3), rng.normal(0, 0.3, 3)])
            g, p = gated(mean, cov), plain(mean, cov)
            if gated._gated(mean, cov):
                fired += 1
                assert p.max() <= 0.01 and g.max() == 0.0
            else:
                passed += 1
                np.testing.assert_array_equal(g, p)
    assert fired > 10 and passed > 10
