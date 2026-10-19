# Tests for the M6 trial runner, using the zero-thrust reference policy so
# outcomes are known in advance from how the scenarios are constructed.

from constellai.evaluation.scenarios import ScenarioConfig
from constellai.evaluation.trials import ZeroPolicy, run_trials, summarize_trials
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv


def _env(cfg):
    return MultiAgentAvoidanceEnv(
        n_agents=cfg.n_agents, mean_motion=cfg.mean_motion, dt=cfg.dt,
        collision_radius_m=cfg.collision_radius_m, max_thrust_mps2=cfg.max_thrust_mps2,
        max_steps=cfg.max_steps,
    )


def test_zero_policy_collides_on_collision_course_scenarios():
    cfg = ScenarioConfig(n_agents=1, p_collision_course=1.0)
    trials = run_trials(_env(cfg), ZeroPolicy(), cfg, n_trials=20)
    assert all(t.collided for t in trials)


def test_zero_policy_survives_near_miss_scenarios():
    cfg = ScenarioConfig(n_agents=1, p_collision_course=0.0)
    trials = run_trials(_env(cfg), ZeroPolicy(), cfg, n_trials=20)
    assert not any(t.collided for t in trials)
    assert all(t.min_separation_m > cfg.collision_radius_m for t in trials)


def test_zero_policy_uses_no_fuel_and_never_saturates():
    cfg = ScenarioConfig(n_agents=3)
    trials = run_trials(_env(cfg), ZeroPolicy(), cfg, n_trials=10)
    assert all(t.mean_fuel_used == 0.0 for t in trials)
    assert all(t.saturation_fraction == 0.0 for t in trials)


def test_same_seed_gives_identical_trials():
    cfg = ScenarioConfig()
    a = run_trials(_env(cfg), ZeroPolicy(), cfg, n_trials=10, seed=4)
    b = run_trials(_env(cfg), ZeroPolicy(), cfg, n_trials=10, seed=4)
    assert a == b


def test_summary_reports_rate_and_interval():
    cfg = ScenarioConfig()
    trials = run_trials(_env(cfg), ZeroPolicy(), cfg, n_trials=40)
    summary = summarize_trials(trials)
    assert summary["n_trials"] == 40
    assert summary["collision_free"] == sum(not t.collided for t in trials)
    assert summary["ci95_low"] <= summary["collision_free_rate"] <= summary["ci95_high"]