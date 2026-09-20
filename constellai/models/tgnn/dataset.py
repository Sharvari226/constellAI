"""Builds forecasting examples: features from an OBSERVATION window,
label from whether a conjunction occurs in a separate, later HORIZON
window. Features and label must never share time samples -- otherwise
the model can "see" the conjunction it's supposed to be forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations

import numpy as np

from constellai.orbital_mechanics.propagation import propagate_series
from constellai.orbital_mechanics.tle import TLERecord


@dataclass(frozen=True)
class PairExample:
    satellite_id_a: str
    satellite_id_b: str
    features: np.ndarray  # shape (T_obs, 4): [dx, dy, dz, separation_km]
    label: int  # 1 if min separation in the horizon window < threshold_km


def _relative_features(states_a, states_b) -> np.ndarray:
    rel = np.array([sa.position_km - sb.position_km for sa, sb in zip(states_a, states_b)])
    sep = np.linalg.norm(rel, axis=1, keepdims=True)
    return np.hstack([rel, sep]).astype(np.float32)


def build_forecast_examples(
    records: list[TLERecord],
    obs_start: datetime,
    obs_end: datetime,
    horizon_end: datetime,
    step: timedelta,
    threshold_km: float,
) -> list[PairExample]:
    """obs_start..obs_end -> features. obs_end..horizon_end -> label only,
    never fed into features."""
    obs_states = {r.satellite_id: propagate_series(r, obs_start, obs_end, step) for r in records}
    horizon_states = {r.satellite_id: propagate_series(r, obs_end, horizon_end, step) for r in records}

    examples: list[PairExample] = []
    for a, b in combinations(records, 2):
        features = _relative_features(obs_states[a.satellite_id], obs_states[b.satellite_id])

        h_a, h_b = horizon_states[a.satellite_id], horizon_states[b.satellite_id]
        horizon_sep = np.linalg.norm(
            [sa.position_km - sb.position_km for sa, sb in zip(h_a, h_b)], axis=1
        )
        label = int(horizon_sep.min() < threshold_km) if len(horizon_sep) else 0

        examples.append(PairExample(a.satellite_id, b.satellite_id, features, label))

    return examples