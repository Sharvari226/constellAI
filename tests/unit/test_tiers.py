# Calibration tests for the M6 scenario difficulty tiers.

import numpy as np
import pytest

from constellai.evaluation.envs import make_env
from constellai.evaluation.scenarios import no_thrust_collision_rate
from constellai.evaluation.tiers import TIERS
from constellai.evaluation.trials import run_trials, summarize_trials


class SteadyDrift:
    # a small constant push on every axis: the cheapest way to defeat slow collision courses
    name = "steady_drift"

    def __init__(self, fraction):
        self.fraction = fraction

    def act(self, observations):
        return [np.ones(3) * self.fraction for _ in observations]


@pytest.mark.parametrize("tier", list(TIERS))
def test_zero_thrust_trial_collision_rate_is_comparable_across_tiers(tier):
    stats = no_thrust_collision_rate(TIERS[tier], n_episodes=200)
    assert 0.30 <= stats["any_agent_collision_rate"] <= 0.60


def test_short_notice_tier_is_harder_for_a_small_steady_drift():
    free = {}
    for tier in ("nominal", "short_notice"):
        cfg = TIERS[tier]
        env = make_env("hcw", cfg, use_filter=False).env
        free[tier] = summarize_trials(run_trials(env, SteadyDrift(0.1), cfg, 100, 0))["collision_free"]
    assert free["short_notice"] <= free["nominal"] - 8


@pytest.mark.parametrize("tier", list(TIERS))
def test_start_positions_are_outside_the_collision_radius(tier):
    from constellai.evaluation.scenarios import STREAM_EVAL, make_rng, sample_scenario

    cfg = TIERS[tier]
    for index in range(30):
        positions, _ = sample_scenario(make_rng(STREAM_EVAL, 0, index), cfg)
        assert all(np.linalg.norm(p) > cfg.collision_radius_m for p in positions)