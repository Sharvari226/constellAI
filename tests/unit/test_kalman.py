import numpy as np
import pytest

from constellai.evaluation.scenarios import reference_mean_motion
from constellai.orbital_mechanics.relative_motion import RelativeState, propagate_hcw_step
from constellai.tracking.kalman import PairFilter, hcw_transition

N = reference_mean_motion()
DT = 10.0
A_MAX = 0.01
SIGMA_A = A_MAX / np.sqrt(3.0)  # std of a uniform(-A_MAX, A_MAX) threat acceleration


@pytest.mark.parametrize("dt", [10.0, -10.0, 5.0])
def test_transition_matches_propagator(dt):
    rng = np.random.default_rng(0)
    phi, b = hcw_transition(N, dt)
    for _ in range(20):
        x = np.concatenate([rng.normal(0, 1000, 3), rng.normal(0, 1, 3)])
        u = rng.uniform(-A_MAX, A_MAX, 3)
        ref = propagate_hcw_step(RelativeState.from_vector(x), u, N, dt).as_vector()
        np.testing.assert_allclose(phi @ x + b @ u, ref, rtol=1e-10, atol=1e-9)


def run_filter(sigma_p, filter_accel_std, n_runs=100, n_steps=100, seed=0):
    """Per-run time-averaged NEES; runs are the independent clusters."""
    phi, b = hcw_transition(N, DT)
    rng = np.random.default_rng(seed)
    out = np.empty(n_runs)
    for r in range(n_runs):
        x = np.concatenate([rng.normal(0, 500, 3), rng.normal(0, 0.5, 3)])
        f = PairFilter(N, DT, sigma_p, filter_accel_std)
        f.init(x, rng)
        vals = []
        for _ in range(n_steps):
            u_ego = rng.uniform(-A_MAX, A_MAX, 3)
            u_thr = rng.uniform(-A_MAX, A_MAX, 3)
            x = phi @ x + b @ (u_thr - u_ego)
            f.predict(-u_ego)
            f.update(x[:3] + sigma_p * rng.standard_normal(3))
            e = x - f.mean
            vals.append(e @ np.linalg.solve(f.cov, e))
        out[r] = np.mean(vals)
    return out


@pytest.mark.parametrize("sigma_p", [5.0, 25.0, 50.0])
def test_nees_consistent(sigma_p):
    nees = run_filter(sigma_p, SIGMA_A)
    se = nees.std(ddof=1) / np.sqrt(len(nees))
    assert abs(nees.mean() - 6.0) < 3.0 * se, (nees.mean(), se)


def test_nees_test_has_power():
    # a filter that ignores process noise must be flagged as overconfident
    assert run_filter(5.0, 0.0).mean() > 20.0


def test_noiseless_limit_converges():
    phi, b = hcw_transition(N, DT)
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0, 500, 3), rng.normal(0, 0.5, 3)])
    f = PairFilter(N, DT, sigma_p=1e-6, process_accel_std=SIGMA_A)
    f.init(x, rng)
    for _ in range(30):
        x = phi @ x
        f.predict(np.zeros(3))
        f.update(x[:3])
    assert np.abs(f.mean - x).max() < 1e-3
