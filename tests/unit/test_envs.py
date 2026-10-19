# Tests for the M6 environment stack: simulator + optional M5 filter + scaling wrapper.

from dataclasses import replace

import numpy as np
import pytest

from constellai.evaluation.envs import SIMULATORS, LiveAgentFilteredEnv, make_env
from constellai.evaluation.scenarios import ScenarioConfig
from constellai.evaluation.tiers import TIERS
from constellai.evaluation.trials import ZeroPolicy, run_trials, summarize_trials
from constellai.models.marl.multi_agent_environment import MultiAgentAvoidanceEnv
from constellai.safety.filtered_environment import FilteredMultiAgentEnv

CFG = ScenarioConfig()


def _raw(n_agents, **overrides):
    kwargs = dict(
        n_agents=n_agents, mean_motion=CFG.mean_motion, dt=CFG.dt,
        collision_radius_m=CFG.collision_radius_m, max_thrust_mps2=CFG.max_thrust_mps2,
        max_steps=CFG.max_steps,
    )
    kwargs.update(overrides)
    return MultiAgentAvoidanceEnv(**kwargs)


@pytest.mark.parametrize("simulator", SIMULATORS)
@pytest.mark.parametrize("use_filter", [False, True])
def test_every_simulator_and_filter_combination_runs(simulator, use_filter):
    built = make_env(simulator, CFG, use_filter)
    obs = built.env.reset(
        [np.array([900.0, 0, 0]), np.array([0, 1200.0, 0]), np.array([0, 0, 1500.0])],
        [np.array([-0.5, 0, 0]), np.zeros(3), np.zeros(3)],
    )
    assert len(obs) == 3
    obs, result = built.env.step([np.zeros(3)] * 3)
    assert len(result.infos) == 3
    assert (built.filtered is not None) == use_filter


def test_unknown_simulator_is_rejected():
    with pytest.raises(ValueError):
        make_env("not_a_simulator", CFG, False)


def test_live_filter_matches_the_parent_filter_while_all_agents_are_alive():
    # generous thrust limit so the filter genuinely corrects the first agent
    positions = [np.array([80.0, 0, 0]), np.array([5000.0, 0, 0])]
    velocities = [np.array([-3.0, 0, 0]), np.zeros(3)]
    parent = FilteredMultiAgentEnv(_raw(2, max_thrust_mps2=1.0))
    live = LiveAgentFilteredEnv(_raw(2, max_thrust_mps2=1.0))
    parent.reset(positions, velocities)
    live.reset(positions, velocities)

    for _ in range(15):
        obs_p, result_p = parent.step([np.zeros(3), np.zeros(3)])
        obs_l, result_l = live.step([np.zeros(3), np.zeros(3)])
        for a, b in zip(obs_p, obs_l):
            np.testing.assert_allclose(a, b, atol=1e-9)
        assert parent.stats.corrected_actions == live.stats.corrected_actions
        assert parent.stats.total_actions == live.stats.total_actions
        if any(result_p.dones):
            break
    assert live.stats.corrected_actions > 0  # the comparison was not vacuous


def test_live_filter_does_not_count_finished_agents():
    positions = [np.array([10.0, 0, 0]), np.array([5000.0, 0, 0])]  # agent 0 collides at once (radius 500)
    velocities = [np.zeros(3), np.zeros(3)]
    parent = FilteredMultiAgentEnv(_raw(2, collision_radius_m=500.0))
    live = LiveAgentFilteredEnv(_raw(2, collision_radius_m=500.0))
    parent.reset(positions, velocities)
    live.reset(positions, velocities)
    for _ in range(2):
        parent.step([np.zeros(3), np.zeros(3)])
        live.step([np.zeros(3), np.zeros(3)])
    assert live.stats.total_actions == 3  # 2 live agents, then 1
    assert parent.stats.total_actions == 4  # the parent keeps counting the finished agent


@pytest.mark.parametrize("tier", list(TIERS))
@pytest.mark.parametrize("simulator", SIMULATORS)
def test_calibrated_gains_let_the_filter_save_a_do_nothing_policy(tier, simulator):
    # single agent on a certain collision course, policy never thrusts: only the filter can act
    cfg = replace(TIERS[tier], n_agents=1, p_collision_course=1.0)
    off = make_env(simulator, cfg, use_filter=False).env
    on = make_env(simulator, cfg, use_filter=True, alpha0=0.03, alpha1=0.03).env
    free_off = summarize_trials(run_trials(off, ZeroPolicy(), cfg, 20, 0))["collision_free"]
    free_on = summarize_trials(run_trials(on, ZeroPolicy(), cfg, 20, 0))["collision_free"]
    assert free_off == 0
    assert free_on >= 18