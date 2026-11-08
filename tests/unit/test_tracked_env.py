import numpy as np

from constellai.evaluation.scaled_env import ScaledMultiAgentEnv
from constellai.evaluation.scenarios import reference_mean_motion
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv
from constellai.tracking.tracker import TrackedMultiAgentEnv

N = reference_mean_motion()
Z = np.zeros(3)
SIGMA_A = 0.01 / np.sqrt(3.0)


def DUMMY(mean, cov):
    return np.full(4, 0.25)


def build(n=2, forecaster=None, noise_seed=0, sigma_p=25.0):
    raw = MultiAgentAvoidanceEnv(n_agents=n, mean_motion=N)
    scaled = ScaledMultiAgentEnv(raw, n, raw.max_thrust_mps2, raw.dt)
    return TrackedMultiAgentEnv(scaled, sigma_p=sigma_p, process_accel_std=SIGMA_A,
                                noise_seed=noise_seed, forecaster=forecaster)


def far_start(n):
    pos = [np.array([1000.0, 0, 0]), np.array([0, 1000.0, 0]), np.array([0, 0, 1000.0])][:n]
    return pos, [Z.copy() for _ in range(n)]


def rollout(env, n, actions_seq, episode_index=0):
    pos, vel = far_start(n)
    obs = [env.reset(pos, vel, episode_index=episode_index)]
    for a in actions_seq:
        obs.append(env.step(a)[0])
    return obs


def test_observation_shapes_and_primary_valid():
    pos, vel = far_start(2)
    a, b = build(), build(forecaster=DUMMY)
    oa, ob = a.reset(pos, vel), b.reset(pos, vel)
    assert all(o.shape == (33,) for o in oa) and all(o.shape == (41,) for o in ob)
    assert all(o[7] == 1.0 for o in oa)  # primary slot valid flag


def test_arm_a_is_exact_prefix_of_arm_b():
    rng = np.random.default_rng(3)
    seq = [[rng.uniform(-1, 1, 3) for _ in range(3)] for _ in range(15)]
    oa = rollout(build(3), 3, seq)
    ob = rollout(build(3, forecaster=DUMMY), 3, seq)
    for ta, tb in zip(oa, ob):
        for xa, xb in zip(ta, tb):
            np.testing.assert_array_equal(xa, xb[:33])
            np.testing.assert_allclose(xb[33:], 0.25)


def test_determinism_and_seed_sensitivity():
    seq = [[np.array([0.3, -0.2, 0.1]) for _ in range(2)] for _ in range(5)]
    o1, o2 = rollout(build(), 2, seq), rollout(build(), 2, seq)
    o3 = rollout(build(noise_seed=1), 2, seq)
    np.testing.assert_array_equal(np.concatenate(o1[-1]), np.concatenate(o2[-1]))
    assert not np.allclose(np.concatenate(o1[-1]), np.concatenate(o3[-1]))


def test_measurement_noise_independent_of_actions():
    e1, e2 = build(), build()
    pos, vel = far_start(2)
    e1.reset(pos, vel, episode_index=4)
    e2.reset(pos, vel, episode_index=4)
    rng = np.random.default_rng(0)
    for _ in range(6):
        e1.step([Z, Z])
        e2.step([rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)])
        np.testing.assert_array_equal(e1.last_noise[0], e2.last_noise[0])
        np.testing.assert_array_equal(e1.last_noise[1], e2.last_noise[1])


def test_done_threat_is_dropped_and_done_ego_is_frozen():
    env = build(3)
    pos = [np.array([1000.0, 0, 0]), np.array([0, 1000.0, 0]), np.array([30.0, 0, 0])]  # agent 2 starts inside the primary's radius
    env.reset(pos, [Z.copy() for _ in range(3)], episode_index=0)
    obs, _ = env.step([Z, Z, Z])
    assert env.raw.live_mask() == [True, True, False]
    assert env.slot1_index(0) == 1 and env.slot1_index(1) == 0
    assert obs[0][20] == 1.0 and obs[1][20] == 1.0
    obs2, _ = env.step([Z, Z, Z])
    np.testing.assert_array_equal(obs2[2], obs[2])


def test_pair_collision_clears_slot1():
    env = build(2)
    a = np.array([1000.0, 0, 0])
    env.reset([a, a + np.array([25.0, 0, 0])], [Z.copy(), Z.copy()], episode_index=0)
    obs, _ = env.step([Z, Z])
    assert env.raw.live_mask() == [False, False]
    assert obs[0][20] == 0.0 and obs[1][20] == 0.0


def test_privileged_state_matches_scaled_layout():
    pos, vel = far_start(2)
    vel = [np.array([0.0, 0.3, 0.0]), np.array([0.2, 0.0, 0.0])]
    raw = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=N)
    ref = np.concatenate(ScaledMultiAgentEnv(raw, 2, raw.max_thrust_mps2, raw.dt).reset(pos, vel))
    env = build(2)
    env.reset(pos, vel)
    got = env.privileged_state().reshape(2, 13)
    np.testing.assert_allclose(got[:, :7].reshape(-1), ref, rtol=1e-12)
    np.testing.assert_allclose(got[:, 7:], 0.0, atol=1e-12)


def test_rewards_and_costs_pass_through_unchanged():
    pos, vel = far_start(2)
    raw = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=N)
    ref = ScaledMultiAgentEnv(raw, 2, raw.max_thrust_mps2, raw.dt)
    ref.reset(pos, vel)
    env = build(2)
    env.reset(pos, vel)
    acts = [np.array([0.5, -0.5, 0.2]), np.array([-0.1, 0.3, 0.0])]
    _, r_ref = ref.step(acts)
    _, r_env = env.step(acts)
    assert r_ref.mission_rewards == r_env.mission_rewards
    assert r_ref.safety_costs == r_env.safety_costs


def test_slot_error_is_observed_and_matches_reward_input():
    env = build(2)
    pos, vel = far_start(2)
    obs = env.reset(pos, vel, episode_index=0)
    np.testing.assert_allclose(obs[0][1:7], 0.0, atol=1e-12)
    for _ in range(25):
        obs, _ = env.step([np.array([1.0, 0.0, 0.0]), Z])
    d_obs = np.linalg.norm(obs[0][1:4]) * 1000.0
    d_raw = np.linalg.norm(env.raw._states[0].position - env.raw._nominal[0].position)
    assert d_raw > 100.0
    np.testing.assert_allclose(d_obs, d_raw, rtol=1e-9)
    np.testing.assert_allclose(obs[1][1:7], 0.0, atol=1e-9)
    priv = env.privileged_state().reshape(2, 13)
    np.testing.assert_allclose(priv[0, 7:], obs[0][1:7], atol=1e-12)


def test_reward_scale_scales_rewards_only():
    def mk(scale):
        raw = MultiAgentAvoidanceEnv(n_agents=2, mean_motion=N)
        scaled = ScaledMultiAgentEnv(raw, 2, raw.max_thrust_mps2, raw.dt)
        return TrackedMultiAgentEnv(scaled, sigma_p=25.0, process_accel_std=0.0,
                                    neighbour_thrust_known=True, reward_scale=scale)
    pos, vel = far_start(2)
    a, b = mk(1.0), mk(0.1)
    a.reset(pos, vel, episode_index=0)
    b.reset(pos, vel, episode_index=0)
    acts = [np.array([0.5, -0.5, 0.2]), np.array([-0.1, 0.3, 0.0])]
    oa, ra = a.step(acts)
    ob, rb = b.step(acts)
    assert min(abs(r) for r in ra.mission_rewards) > 0.0
    np.testing.assert_allclose(np.array(rb.mission_rewards), 0.1 * np.array(ra.mission_rewards), rtol=1e-12)
    assert rb.safety_costs == ra.safety_costs
    np.testing.assert_array_equal(np.concatenate(oa), np.concatenate(ob))
