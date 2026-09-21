"""M3->M4 wiring: wraps SingleAgentAvoidanceEnv so its observation
includes M3's predicted risk (mean, std) alongside the existing
physics state -- observation grows from 7-dim to 9-dim. safety_cost
and the environment's own physics are completely untouched; see
risk_adapter.py's module docstring for why observation-only was the
correct integration point.
"""

from __future__ import annotations

import numpy as np

from constellai.models.marl.environment import SingleAgentAvoidanceEnv, StepResult
from constellai.models.tgnn.risk_adapter import TGNNRiskAdapter


class ForecastAugmentedSingleAgentEnv:
    """Wraps SingleAgentAvoidanceEnv, appending [predicted_risk_mean,
    predicted_risk_std] to every observation. Composition, not a fork --
    matches the project's established wrapping pattern (see
    FilteredSingleAgentEnv)."""

    def __init__(self, env: SingleAgentAvoidanceEnv, risk_adapter: TGNNRiskAdapter):
        self.env = env
        self.risk_adapter = risk_adapter

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        rel_position, rel_velocity = obs[:3], obs[3:6]
        mean_risk, std_risk = self.risk_adapter.predict_risk(rel_position, rel_velocity)
        return np.concatenate([obs, [mean_risk, std_risk]]).astype(np.float32)

    def reset(self, initial_position_m: np.ndarray, initial_velocity_mps: np.ndarray) -> np.ndarray:
        obs = self.env.reset(initial_position_m, initial_velocity_mps)
        return self._augment(obs)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, StepResult]:
        obs, result = self.env.step(action)
        return self._augment(obs), result