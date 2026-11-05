
import numpy as np

from constellai.evaluation.scenarios import (
    STREAM_CAL, STREAM_EVAL, STREAM_FC_TRAIN, STREAM_FC_VAL, STREAM_TRAIN,
    make_rng, reference_mean_motion, warning_config,
)
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv
from constellai.tracking.kalman import hcw_transition

N = reference_mean_motion()
Z = np.zeros(3)


def make():
    return MultiAgentAvoidanceEnv(n_agents=2, mean_motion=N)


def far_apart(env, v0=Z, v1=Z):
    env.reset([np.array([1000.0, 0, 0]), np.array([0, 1000.0, 0])], [v0.copy(), v1.copy()])


def test_applied_thrust_is_exposed_and_clipped():
    env = make()
    far_apart(env)
    _, res = env.step([np.array([5.0, -5.0, 0.0]), Z])
    lim = env.max_thrust_mps2
    np.testing.assert_allclose(res.infos[0]["thrust_mps2"], [lim, -lim, 0.0])
    np.testing.assert_allclose(res.infos[1]["thrust_mps2"], Z)


def test_accessors_match_reset_state():
    env = make()
    far_apart(env, v0=np.array([0.0, 0.3, 0.0]))
    x = env.agent_state_vectors()
    np.testing.assert_allclose(x[0], [1000.0, 0, 0, 0.0, 0.3, 0.0])
    assert env.live_mask() == [True, True]
    assert env.fuel_remaining() == [env.fuel_budget] * 2


def test_live_mask_after_pair_collision():
    env = make()
    a = np.array([1000.0, 0, 0])
    env.reset([a, a + np.array([25.0, 0, 0])], [Z.copy(), Z.copy()])
    env.step([Z, Z])
    assert env.live_mask() == [False, False]


def test_env_relative_dynamics_match_filter_model():
    # the Kalman filter assumes: rel' = Phi rel + B (u_threat - u_ego), primary' = Phi p + B(-u_ego)
    env = make()
    far_apart(env, v0=np.array([0.0, 0.3, 0.0]), v1=np.array([0.2, 0.0, 0.0]))
    x0 = env.agent_state_vectors()
    _, res = env.step([np.array([0.004, -0.002, 0.001]), np.array([-0.003, 0.005, 0.0])])
    x1 = env.agent_state_vectors()
    u0, u1 = res.infos[0]["thrust_mps2"], res.infos[1]["thrust_mps2"]
    phi, b = hcw_transition(N, env.dt)
    np.testing.assert_allclose(x1[1] - x1[0], phi @ (x0[1] - x0[0]) + b @ (u1 - u0), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(-x1[0], phi @ (-x0[0]) + b @ (-u0), rtol=1e-9, atol=1e-9)


def test_streams_and_presets():
    streams = [STREAM_TRAIN, STREAM_EVAL, STREAM_FC_TRAIN, STREAM_FC_VAL, STREAM_CAL]
    assert len(set(streams)) == 5
    assert make_rng(STREAM_FC_TRAIN, 0, 0).random() != make_rng(STREAM_TRAIN, 0, 0).random()
    assert warning_config("short").time_to_tca_s == (150.0, 600.0)
    assert warning_config("long").time_to_tca_s == (500.0, 1500.0)
