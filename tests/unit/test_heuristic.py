import numpy as np

from constellai.evaluation.heuristic import SLOT0, SLOT1, ThresholdHeuristicPolicy, closest_approach
from constellai.evaluation.scaled_env import ScaledMultiAgentEnv
from constellai.evaluation.scenarios import STREAM_FC_VAL, make_rng, sample_scenario, warning_config
from constellai.evaluation.trials import run_trials
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv
from constellai.tracking.tracker import TrackedMultiAgentEnv

RC = 50.0


def obs_with(slot0=None, slot1=None):
    o = np.zeros(33)
    for start, s in ((SLOT0, slot0), (SLOT1, slot1)):
        if s is not None:
            pos, vel = s
            o[start] = 1.0
            o[start + 1:start + 4] = np.asarray(pos, dtype=float) / 1000.0
            o[start + 4:start + 7] = vel
    return o


def policy(tau=2.0, **kw):
    return ThresholdHeuristicPolicy(tau, RC, **kw)


def test_closest_approach_known_pass():
    miss, m = closest_approach(np.array([500.0, 20.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 600.0)
    assert abs(miss - 20.0) < 1e-9 and np.allclose(m, [0.0, 20.0, 0.0])


def test_burns_away_from_predicted_closest_approach():
    o = obs_with(slot0=([500.0, 20.0, 0.0], [-1.0, 0.0, 0.0]))
    np.testing.assert_allclose(policy().act([o])[0], [0.0, -1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(policy(burn_fraction=0.5).act([o])[0], [0.0, -0.5, 0.0], atol=1e-12)


def test_coasts_when_miss_large_beyond_horizon_or_invalid():
    far = obs_with(slot0=([500.0, 300.0, 0.0], [-1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(policy().act([far])[0], np.zeros(3))
    late = obs_with(slot0=([5000.0, 20.0, 0.0], [-1.0, 0.0, 0.0]))  # closest approach at 5000 s, beyond 600 s
    np.testing.assert_array_equal(policy().act([late])[0], np.zeros(3))
    invalid = obs_with(slot0=([500.0, 20.0, 0.0], [-1.0, 0.0, 0.0]))
    invalid[SLOT0] = 0.0
    np.testing.assert_array_equal(policy().act([invalid])[0], np.zeros(3))


def test_acts_on_most_threatening_slot():
    o = obs_with(slot0=([500.0, 60.0, 0.0], [-1.0, 0.0, 0.0]), slot1=([400.0, 0.0, -10.0], [-1.0, 0.0, 0.0]))
    np.testing.assert_allclose(policy().act([o])[0], [0.0, 0.0, 1.0], atol=1e-12)


def test_exactly_head_on_steps_sideways():
    o = obs_with(slot0=([500.0, 0.0, 0.0], [-1.0, 0.0, 0.0]))
    a = policy().act([o])[0]
    np.testing.assert_allclose(a, [0.0, 0.0, -1.0], atol=1e-12)
    assert abs(a @ np.array([-1.0, 0.0, 0.0])) < 1e-12  # perpendicular to the relative velocity


def test_runs_in_tracked_env_and_obs_layout_matches():
    cfg = warning_config("short")
    raw = MultiAgentAvoidanceEnv(n_agents=cfg.n_agents, mean_motion=cfg.mean_motion, dt=cfg.dt,
                                 collision_radius_m=cfg.collision_radius_m,
                                 max_thrust_mps2=cfg.max_thrust_mps2, max_steps=cfg.max_steps)
    scaled = ScaledMultiAgentEnv(raw, cfg.n_agents, cfg.max_thrust_mps2, cfg.dt)
    env = TrackedMultiAgentEnv(scaled, sigma_p=5.0, process_accel_std=0.0, neighbour_thrust_known=True)
    pos, vel = sample_scenario(make_rng(STREAM_FC_VAL, 0, 0), cfg)
    obs = env.reset(pos, vel, episode_index=0)
    assert all(o[SLOT0] == 1.0 for o in obs) and all(o[SLOT1] == 1.0 for o in obs)
    trials = run_trials(env, ThresholdHeuristicPolicy(2.0, cfg.collision_radius_m), cfg, 3,
                        seed=0, stream=STREAM_FC_VAL)
    assert len(trials) == 3
