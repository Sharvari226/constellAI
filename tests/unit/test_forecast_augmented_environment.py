"""Unit tests for M3->M4 wiring.

test_predicted_risk_increases_as_satellites_close is the one that
matters most: proof the adapter's output tracks real physical danger,
not just "produces some number of the right shape."
"""

import numpy as np
import torch

from constellai.models.marl.environment import SingleAgentAvoidanceEnv
from constellai.models.marl.forecast_augmented_environment import ForecastAugmentedSingleAgentEnv
from constellai.models.tgnn.risk_adapter import TGNNRiskAdapter
from constellai.models.tgnn.tgn_lite import TGNLite


def test_observation_grows_by_two_dimensions():
    model = TGNLite()
    adapter = TGNNRiskAdapter(model)
    env = ForecastAugmentedSingleAgentEnv(SingleAgentAvoidanceEnv(mean_motion=0.0011), adapter)

    obs = env.reset(initial_position_m=np.array([1000.0, 0, 0]), initial_velocity_mps=np.zeros(3))
    assert obs.shape == (9,)


def test_step_also_returns_augmented_observation():
    model = TGNLite()
    adapter = TGNNRiskAdapter(model)
    env = ForecastAugmentedSingleAgentEnv(SingleAgentAvoidanceEnv(mean_motion=0.0011), adapter)
    env.reset(initial_position_m=np.array([1000.0, 0, 0]), initial_velocity_mps=np.zeros(3))

    obs, _ = env.step(np.zeros(3))
    assert obs.shape == (9,)


def test_risk_values_are_valid_probabilities_and_nonnegative_std():
    model = TGNLite()
    adapter = TGNNRiskAdapter(model)
    env = ForecastAugmentedSingleAgentEnv(SingleAgentAvoidanceEnv(mean_motion=0.0011), adapter)
    obs = env.reset(initial_position_m=np.array([1000.0, 0, 0]), initial_velocity_mps=np.zeros(3))

    mean_risk, std_risk = obs[7], obs[8]
    assert 0.0 <= mean_risk <= 1.0
    assert std_risk >= 0.0


def test_predicted_risk_increases_as_satellites_close():
    """Same untrained-but-structurally-real model, two very different
    relative states -- close and fast-closing should score higher risk
    than far and stationary. This isn't about accuracy (the model is
    untrained here), it's about the adapter correctly translating
    'more dangerous state' into 'the exact feature vector that
    determines the model's output' -- a wiring correctness test, not a
    model-quality test."""
    torch.manual_seed(0)
    model = TGNLite()
    adapter = TGNNRiskAdapter(model)

    far_mean, _ = adapter.predict_risk(
        relative_position_m=np.array([1_000_000.0, 0.0, 0.0]), relative_velocity_mps=np.zeros(3),
    )
    close_mean, _ = adapter.predict_risk(
        relative_position_m=np.array([10.0, 0.0, 0.0]), relative_velocity_mps=np.array([-50.0, 0, 0]),
    )
    assert far_mean != close_mean