"""Conjunction geometry: time of closest approach (TCA) and miss distance.

This module is the shared dependency for both the Step-3 non-ML baseline
(threshold on miss distance) and M2's sparse graph edge construction
(closing rate + projected miss distance + TCA, not raw Euclidean distance
at a single instant). Keeping this logic in one place means the baseline
and the graph-construction module can never quietly disagree about what
"close" means.

All geometry here operates on already-propagated StateVector pairs in a
common frame (TEME, matching propagation.py's native output) -- this
module does no propagation itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from constellai.orbital_mechanics.propagation import StateVector


@dataclass(frozen=True)
class ConjunctionEvent:
    """Result of screening one satellite pair over a time window.

    Attributes
    ----------
    satellite_id_a, satellite_id_b : str
        NORAD IDs of the pair, in the order screened.
    tca : datetime
        Time of closest approach found within the screened window.
    miss_distance_km : float
        Minimum separation at TCA, kilometers.
    relative_speed_km_s : float
        Magnitude of relative velocity at TCA -- the "closing rate" used
        by M2's graph construction, not just raw distance.
    """

    satellite_id_a: str
    satellite_id_b: str
    tca: datetime
    miss_distance_km: float
    relative_speed_km_s: float


def _separation_km(a: StateVector, b: StateVector) -> float:
    return float(np.linalg.norm(a.position_km - b.position_km))


def _relative_speed_km_s(a: StateVector, b: StateVector) -> float:
    return float(np.linalg.norm(a.velocity_km_s - b.velocity_km_s))


def screen_pair(
    states_a: list[StateVector],
    states_b: list[StateVector],
) -> ConjunctionEvent:
    """Find TCA and miss distance for one satellite pair over pre-propagated
    state series.

    Parameters
    ----------
    states_a, states_b : list[StateVector]
        Propagated states for each satellite, at matching timestamps
        (same length, same epoch at each index) -- callers are
        responsible for propagating both satellites over the same
        sample grid (see propagate_series in propagation.py). This
        function does not itself refine between samples; the sample
        interval passed to propagate_series determines TCA precision.

    Returns
    -------
    ConjunctionEvent
        The minimum-separation event found across the sampled window.

    Raises
    ------
    ValueError
        If the two state series have mismatched lengths or timestamps,
        or if either series is empty.
    """
    if len(states_a) != len(states_b) or len(states_a) == 0:
        raise ValueError(
            "screen_pair requires two non-empty, equal-length, "
            "time-aligned state series"
        )

    separations = [_separation_km(a, b) for a, b in zip(states_a, states_b)]
    min_idx = int(np.argmin(separations))

    a_at_min, b_at_min = states_a[min_idx], states_b[min_idx]
    if a_at_min.epoch != b_at_min.epoch:
        raise ValueError(
            f"Time-alignment violated at index {min_idx}: "
            f"{a_at_min.epoch} != {b_at_min.epoch}"
        )

    return ConjunctionEvent(
        satellite_id_a=a_at_min.satellite_id,
        satellite_id_b=b_at_min.satellite_id,
        tca=a_at_min.epoch,
        miss_distance_km=separations[min_idx],
        relative_speed_km_s=_relative_speed_km_s(a_at_min, b_at_min),
    )