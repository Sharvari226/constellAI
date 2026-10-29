import numpy as np
from constellai.models.marl.objective import (
    ObjectiveConfig, predicted_miss, shaped_risk, reward, costs, PIDLagrange)

cfg = ObjectiveConfig()
Z = np.zeros(3)

def test_shaped_risk_bounds():
    assert shaped_risk(cfg.d_warn + 1, cfg) == 0.0
    assert shaped_risk(cfg.d_safe - 0.1, cfg) == 1.0

def test_reward_zero_at_rest():
    assert reward(Z, Z, Z, cfg) == 0.0

def test_predicted_miss_straight_pass():
    # closest approach of a straight-line pass is the perpendicular offset
    d = predicted_miss([-10.0, 2.0, 0.0], [1.0, 0.0, 0.0], horizon=100.0)
    assert abs(d - 2.0) < 1e-9

def test_secondary_cost_needs_burn():
    near = [(np.array([2.0, 0, 0]), Z)]
    idle = costs(Z, Z, Z, [], near, cfg)
    burn = costs(np.array([0.5, 0, 0]), Z, Z, [], near, cfg)
    assert idle["sec"] == 0.0 and burn["sec"] > 0.0

def test_filter_cost_zero_when_unchanged():
    a = np.array([0.3, 0, 0])
    assert costs(a, a, a, [], [], cfg)["filter"] == 0.0

def test_lambda_rises_then_falls():
    p = PIDLagrange(budget=0.1)
    up = [p.update(1.0) for _ in range(5)]
    assert up[-1] > up[0] > 0
    down = [p.update(0.0) for _ in range(40)]
    assert down[-1] < up[-1]

def test_no_derivative_kick_on_first_update():
    p = PIDLagrange(budget=0.1)
    assert abs(p.update(1.0) - (0.5*0.9 + 0.05*0.9)) < 1e-12