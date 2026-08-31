"""M2, part B: fine relative-dynamics screening for graph edge inclusion.

Takes coarse-filter survivors and decides which become graph edges,
reusing the same TCA/miss-distance/closing-rate logic as the Step-3
baseline (conjunction.screen_pair) -- not a second, inconsistent notion
of risk. Edge inclusion requires BOTH miss distance below a threshold
AND closing rate above a floor -- a pair close but separating fast is
not an edge; a pair far but closing fast still is. That's what "relative
dynamics, not raw distance" means in code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from constellai.common.constants import (
    DEFAULT_SCREENING_DISTANCE_KM,
    DEFAULT_MIN_CLOSING_RATE_KM_S,
)
from constellai.orbital_mechanics.conjunction import ConjunctionEvent, screen_pair
from constellai.orbital_mechanics.propagation import propagate_series
from constellai.orbital_mechanics.tle import TLERecord


@dataclass(frozen=True)
class GraphEdge:
    """One edge in M2's sparse dynamic graph -- an edge IS a
    ConjunctionEvent that passed the relative-dynamics test."""

    event: ConjunctionEvent

    @property
    def satellite_ids(self) -> tuple[str, str]:
        return (self.event.satellite_id_a, self.event.satellite_id_b)


def screen_candidate_pair(
    record_a: TLERecord,
    record_b: TLERecord,
    start: datetime,
    end: datetime,
    step: timedelta,
    distance_threshold_km: float = DEFAULT_SCREENING_DISTANCE_KM,
    min_closing_rate_km_s: float = DEFAULT_MIN_CLOSING_RATE_KM_S,
) -> GraphEdge | None:
    """Fine-screen one coarse-filter-survivor pair.

    Returns None (not an empty/sentinel edge) if the pair fails either
    condition -- so callers can't accidentally treat a rejected pair as
    a zero-risk edge.
    """
    states_a = propagate_series(record_a, start, end, step)
    states_b = propagate_series(record_b, start, end, step)

    event = screen_pair(states_a, states_b)

    if event.miss_distance_km >= distance_threshold_km:
        return None
    if event.relative_speed_km_s < min_closing_rate_km_s:
        return None

    return GraphEdge(event=event)