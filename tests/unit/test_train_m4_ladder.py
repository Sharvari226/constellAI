# Smoke tests for the M4 training script: every rung must run end to end on a
# tiny budget and produce a history; the same seed must reproduce the same run.

import pytest

from constellai.evaluation.scenarios import ScenarioConfig
from constellai.models.marl.checkpoint import RUNG_NAMES
from constellai.scripts.train_m4_ladder import train_rung


def _tiny_cfg():
    # 40 steps x 3 episodes gives MADDPG's replay buffer more than its
    # 64-transition minimum, so its update path is exercised too
    return ScenarioConfig(n_agents=3, max_steps=40)


@pytest.mark.parametrize("rung", RUNG_NAMES)
def test_every_rung_trains_and_returns_history(rung):
    result = train_rung(
        rung, _tiny_cfg(), seed=0, episodes=3, maddpg_updates=1, log_every=0,
    )
    assert len(result.history) == 3
    assert all("any_collision" in row for row in result.history)
    assert len(result.networks) >= 1


def test_same_seed_reproduces_the_same_run():
    run_a = train_rung("ippo", _tiny_cfg(), seed=5, episodes=3, log_every=0)
    run_b = train_rung("ippo", _tiny_cfg(), seed=5, episodes=3, log_every=0)
    assert [r["policy_loss"] for r in run_a.history] == [r["policy_loss"] for r in run_b.history]


def test_constrained_rung_reports_lambda():
    result = train_rung("constrained_mappo", _tiny_cfg(), seed=0, episodes=3, log_every=0)
    assert "lambda" in result.history[-1]
    assert result.extra_meta["final_lambda"] >= 0.0


def test_unknown_rung_is_rejected():
    with pytest.raises(ValueError):
        train_rung("nope", _tiny_cfg(), seed=0, episodes=1)