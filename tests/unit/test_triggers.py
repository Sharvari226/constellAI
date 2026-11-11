import numpy as np
import pytest

from constellai.evaluation.heuristic import SLOT0, SLOT1, ThresholdHeuristicPolicy, closest_approach
from constellai.evaluation.scaled_env import ScaledMultiAgentEnv
from constellai.evaluation.scenarios import STREAM_FC_VAL, warning_config
from constellai.evaluation.trials import ZeroPolicy, run_trials, summarize_trials
from constellai.evaluation.triggers import (
    FC0, FC1, DistanceTriggerPolicy, ProbabilityTriggerPolicy, away_direction,
)
from constellai.forecasting.events import AnalyticForecaster
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv
from constellai.tracking.tracker import TrackedMultiAgentEnv

RC = 50.0


def obs41(slot0=None, slot1=None, fc0=(0, 0, 0, 0), fc1=(0, 0, 0, 0)):
    o = np.zeros(41)
    for start, s in ((SLOT0, slot0), (SLOT1, slot1)):
        if s is not None:
            pos, vel = s
            o[start] = 1.0
            o[start + 1:start + 4] = np.asarray(pos, dtype=float) / 1000.0
            o[start + 4:start + 7] = vel
    o[FC0:FC0 + 4] = fc0
    o[FC1:FC1 + 4] = fc1
    return o


def test_direction_rule_matches_threshold_heuristic():
    rng = np.random.default_rng(0)
    cases = [(np.array([500.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]))]  # exactly head-on
    for _ in range(40):
        cases.append((rng.normal(0, 400, 3), rng.normal(0, 0.6, 3)))
    h = ThresholdHeuristicPolicy(1e6, RC)  # trigger radius so large that it always acts
    for pos, vel in cases:
        _, m = closest_approach(pos, vel, 600.0)
        np.testing.assert_allclose(h.act([obs41(slot0=(pos, vel))])[0], away_direction(m, vel), atol=1e-12)


def test_distance_trigger_behaviour():
    pol = DistanceTriggerPolicy(4.0, RC)  # acts inside 200 m
    far = obs41(slot0=([500.0, 0, 0], [-1.0, 0, 0]))
    np.testing.assert_array_equal(pol.act([far])[0], np.zeros(3))
    near = obs41(slot0=([120.0, 0, 90.0], [-0.5, 0, 0]))  # 150 m away
    np.testing.assert_allclose(pol.act([near])[0], -np.array([120.0, 0, 90.0]) / 150.0, atol=1e-12)
    closer = obs41(slot0=([120.0, 0, 90.0], [-0.5, 0, 0]), slot1=([0.0, 60.0, 0.0], [0, 0.1, 0]))
    np.testing.assert_allclose(pol.act([closer])[0], [0.0, -1.0, 0.0], atol=1e-12)
    invalid = near.copy()
    invalid[SLOT0] = 0.0
    np.testing.assert_array_equal(pol.act([invalid])[0], np.zeros(3))


def test_probability_trigger_behaviour():
    pol = ProbabilityTriggerPolicy(0.1, 3)
    pos, vel = [500.0, 20.0, 0.0], [-1.0, 0.0, 0.0]
    low = obs41(slot0=(pos, vel), fc0=(0, 0, 0, 0.05))
    np.testing.assert_array_equal(pol.act([low])[0], np.zeros(3))
    high = obs41(slot0=(pos, vel), fc0=(0, 0, 0, 0.30))
    np.testing.assert_allclose(pol.act([high])[0], [0.0, -1.0, 0.0], atol=1e-12)
    other = obs41(slot0=(pos, vel), fc0=(0, 0.9, 0, 0.0))  # high value at horizon 1 only
    np.testing.assert_array_equal(pol.act([other])[0], np.zeros(3))
    assert np.abs(ProbabilityTriggerPolicy(0.1, 1).act([other])[0]).sum() > 0
    two = obs41(slot0=([500.0, 20.0, 0.0], [-1.0, 0, 0]), slot1=([400.0, 0.0, -10.0], [-1.0, 0, 0]),
                fc0=(0, 0, 0, 0.2), fc1=(0, 0, 0, 0.6))
    np.testing.assert_allclose(pol.act([two])[0], [0.0, 0.0, 1.0], atol=1e-12)
    with pytest.raises(ValueError):
        pol.act([np.zeros(33)])


def test_oracle_trigger_runs_and_is_frugal():
    cfg = warning_config("short")
    raw = MultiAgentAvoidanceEnv(n_agents=cfg.n_agents, mean_motion=cfg.mean_motion, dt=cfg.dt,
                                 collision_radius_m=cfg.collision_radius_m,
                                 max_thrust_mps2=cfg.max_thrust_mps2, max_steps=cfg.max_steps)
    scaled = ScaledMultiAgentEnv(raw, cfg.n_agents, cfg.max_thrust_mps2, cfg.dt)
    oracle = AnalyticForecaster(cfg.mean_motion, cfg.dt, cfg.collision_radius_m, n_samples=1)
    env = TrackedMultiAgentEnv(scaled, sigma_p=5.0, process_accel_std=0.0, neighbour_thrust_known=True, oracle=oracle)
    zero = summarize_trials(run_trials(env, ZeroPolicy(), cfg, 20, seed=0, stream=STREAM_FC_VAL))
    trig = summarize_trials(run_trials(env, ProbabilityTriggerPolicy(0.5, 3, kind="oracle"), cfg, 20,
                                       seed=0, stream=STREAM_FC_VAL))
    assert trig["collision_free_rate"] >= zero["collision_free_rate"]
    assert trig["mean_fuel_used_per_agent"] < 5.0


def test_hcw_trigger_differs_from_straight_line_and_shares_direction():
    from constellai.evaluation.scenarios import reference_mean_motion
    from constellai.evaluation.triggers import HcwMissTriggerPolicy
    from constellai.forecasting.events import FirstEntryEngine
    engine = FirstEntryEngine(reference_mean_motion(), 10.0, RC, 60)
    hcw, lin = HcwMissTriggerPolicy(1.2, engine), ThresholdHeuristicPolicy(1.2, RC)
    rng = np.random.default_rng(0)
    only_hcw = only_lin = both = 0
    for _ in range(3000):
        pos, vel = rng.normal(0, 400, 3), rng.normal(0, 0.5, 3)
        o = obs41(slot0=(pos, vel))
        a_h, a_l = hcw.act([o])[0], lin.act([o])[0]
        h_on, l_on = bool(np.abs(a_h).sum() > 0), bool(np.abs(a_l).sum() > 0)
        if h_on and l_on:
            both += 1
            np.testing.assert_allclose(a_h, a_l, atol=1e-12)
        only_hcw += h_on and not l_on
        only_lin += l_on and not h_on
    assert only_hcw > 0 and only_lin > 0 and both > 0


def test_monotone_miscalibration_equals_threshold_shift():
    rng = np.random.default_rng(0)
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    logit = lambda p: np.log(p / (1.0 - p))
    for a, b in ((2.0, 0.0), (0.5, 0.0), (1.0, 1.0), (1.0, -1.0)):
        for pstar in (0.0033, 0.03, 0.11):
            p_eff = float(sig((logit(pstar) - b) / a))
            for _ in range(200):
                pos, vel = rng.normal(0, 400, 3), rng.normal(0, 0.5, 3)
                f0, f1 = np.sort(rng.uniform(0.001, 0.999, 4)), np.sort(rng.uniform(0.001, 0.999, 4))
                base = obs41(slot0=(pos, vel), slot1=(pos[::-1], vel[::-1]), fc0=tuple(f0), fc1=tuple(f1))
                warped = obs41(slot0=(pos, vel), slot1=(pos[::-1], vel[::-1]),
                               fc0=tuple(sig(a * logit(f0) + b)), fc1=tuple(sig(a * logit(f1) + b)))
                act_w = ProbabilityTriggerPolicy(pstar, 3).act([warped])[0]
                act_b = ProbabilityTriggerPolicy(p_eff, 3).act([base])[0]
                np.testing.assert_allclose(act_w, act_b, atol=1e-12)
