# Tests for the shared M4/M6 scenario sampler.
#
# test_collision_course_agents_collide_without_thrust and
# test_near_miss_agents_do_not_collide_without_thrust are the important
# ones: they prove the "built backwards from TCA" construction does what
# it claims, instead of just producing random-looking numbers.

import numpy as np
import pytest

from constellai.evaluation.scenarios import (
    STREAM_EVAL,
    STREAM_TRAIN,
    ScenarioConfig,
    make_rng,
    no_thrust_collision_rate,
    sample_scenario,
)


def test_same_seed_and_index_reproduce_the_scenario():
    cfg = ScenarioConfig()
    p1, v1 = sample_scenario(make_rng(STREAM_EVAL, 3, 11), cfg)
    p2, v2 = sample_scenario(make_rng(STREAM_EVAL, 3, 11), cfg)
    for a, b in zip(p1 + v1, p2 + v2):
        np.testing.assert_array_equal(a, b)


def test_train_and_eval_streams_produce_different_scenarios():
    cfg = ScenarioConfig()
    p_train, _ = sample_scenario(make_rng(STREAM_TRAIN, 0, 0), cfg)
    p_eval, _ = sample_scenario(make_rng(STREAM_EVAL, 0, 0), cfg)
    assert not np.allclose(p_train[0], p_eval[0])


def test_agents_start_separated_from_each_other_and_the_origin():
    cfg = ScenarioConfig()
    min_sep = cfg.min_start_separation_radii * cfg.collision_radius_m
    for index in range(50):
        positions, _ = sample_scenario(make_rng(STREAM_EVAL, 0, index), cfg)
        assert all(np.linalg.norm(p) >= min_sep for p in positions)
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                assert np.linalg.norm(positions[i] - positions[j]) >= min_sep


def test_collision_course_agents_collide_without_thrust():
    cfg = ScenarioConfig(n_agents=1, p_collision_course=1.0)
    stats = no_thrust_collision_rate(cfg, n_episodes=30)
    assert stats["per_agent_collision_rate"] == 1.0


def test_near_miss_agents_do_not_collide_without_thrust():
    cfg = ScenarioConfig(n_agents=1, p_collision_course=0.0)
    stats = no_thrust_collision_rate(cfg, n_episodes=30)
    assert stats["per_agent_collision_rate"] == 0.0


def test_default_distribution_is_dangerous_but_not_hopeless():
    # target band for the zero-thrust TRIAL collision rate (any agent collides)
    stats = no_thrust_collision_rate(ScenarioConfig(), n_episodes=200)
    assert 0.30 <= stats["any_agent_collision_rate"] <= 0.60


def test_unseparable_config_raises():
    cfg = ScenarioConfig(min_start_separation_radii=1e6, max_resample_attempts=3)
    with pytest.raises(RuntimeError):
        sample_scenario(make_rng(STREAM_EVAL, 0, 0), cfg)