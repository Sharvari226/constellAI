"""Fine relative-dynamics screening for graph edge inclusion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from constellai.common.constants import (
    DEFAULT_MIN_CLOSING_RATE_KM_S,
    DEFAULT_SCREENING_DISTANCE_KM,
)
from constellai.orbital_mechanics.conjunction import ConjunctionEvent, screen_pair
from constellai.orbital_mechanics.propagation import propagate_series
from constellai.orbital_mechanics.tle import TLERecord


@dataclass(frozen=True)
class GraphEdge:
    """A conjunction event that passed the graph screening thresholds."""

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
    """Return an edge only when miss distance and closing rate pass."""
    event = screen_pair(
        propagate_series(record_a, start, end, step),
        propagate_series(record_b, start, end, step),
    )
    if event.miss_distance_km >= distance_threshold_km:
        return None
    if event.relative_speed_km_s < min_closing_rate_km_s:
        return None
    return GraphEdge(event=event)