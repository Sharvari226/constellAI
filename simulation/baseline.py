"""Step 3: non-ML conjunction baseline.

Deliberately the simplest thing that could work: propagate every
satellite over the screening window, check every pair exhaustively
(O(N^2) -- no sparsity, no learning), and flag any pair whose miss
distance drops below a fixed threshold.

This is the baseline every downstream model (M3's TGNN, M4's RL) has
to beat -- recording its numbers now, before any ML component exists,
keeps that comparison honest. It is also the "exhaustive pairwise" side
of M2's later false-negative gate: once the sparse graph exists, this
same exhaustive screen runs on the same scenario and the two flagged
sets get diffed against each other.
"""

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
    """Everything the baseline found, plus what it cost to find it."""

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
    """Exhaustively screen every satellite pair for conjunctions."""
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
