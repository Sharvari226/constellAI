"""Conjunction geometry for pre-propagated state series."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .propagation import StateVector


@dataclass(frozen=True)
class ConjunctionEvent:
    """Minimum-separation event found for one satellite pair."""

    satellite_id_a: str
    satellite_id_b: str
    tca: datetime
    miss_distance_km: float
    relative_speed_km_s: float


def screen_pair(
    states_a: list[StateVector],
    states_b: list[StateVector],
) -> ConjunctionEvent:
    """Find the sampled time of closest approach for two state series."""
    if len(states_a) != len(states_b) or not states_a:
        raise ValueError(
            "screen_pair requires two non-empty, equal-length, "
            "time-aligned state series"
        )

    for index, (state_a, state_b) in enumerate(zip(states_a, states_b)):
        if state_a.epoch != state_b.epoch:
            raise ValueError(
                f"Time-alignment violated at index {index}: "
                f"{state_a.epoch} != {state_b.epoch}"
            )

    separations = [
        float(np.linalg.norm(state_a.position_km - state_b.position_km))
        for state_a, state_b in zip(states_a, states_b)
    ]
    min_index = int(np.argmin(separations))
    state_a = states_a[min_index]
    state_b = states_b[min_index]

    return ConjunctionEvent(
        satellite_id_a=state_a.satellite_id,
        satellite_id_b=state_b.satellite_id,
        tca=state_a.epoch,
        miss_distance_km=separations[min_index],
        relative_speed_km_s=float(
            np.linalg.norm(state_a.velocity_km_s - state_b.velocity_km_s)
        ),
    )