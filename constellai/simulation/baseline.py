"""Exhaustive non-ML conjunction screening baseline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations

from constellai.common.constants import DEFAULT_SCREENING_DISTANCE_KM
from constellai.orbital_mechanics.conjunction import ConjunctionEvent, screen_pair
from constellai.orbital_mechanics.propagation import propagate_series
from constellai.orbital_mechanics.tle import TLERecord


@dataclass(frozen=True)
class BaselineResult:
    """Conjunctions found and screening metadata."""

    flagged: list[ConjunctionEvent]
    pairs_screened: int
    threshold_km: float


def run_baseline(
    records: list[TLERecord],
    start: datetime,
    end: datetime,
    step: timedelta,
    threshold_km: float = DEFAULT_SCREENING_DISTANCE_KM,
) -> BaselineResult:
    """Propagate every record and screen every unique satellite pair."""
    if len(records) < 2:
        raise ValueError("run_baseline requires at least two satellites")

    all_states = {
        record.satellite_id: propagate_series(record, start, end, step)
        for record in records
    }
    flagged: list[ConjunctionEvent] = []
    pairs_screened = 0

    for record_a, record_b in combinations(records, 2):
        pairs_screened += 1
        event = screen_pair(
            all_states[record_a.satellite_id],
            all_states[record_b.satellite_id],
        )
        if event.miss_distance_km < threshold_km:
            flagged.append(event)

    return BaselineResult(
        flagged=flagged,
        pairs_screened=pairs_screened,
        threshold_km=threshold_km,
    )