"""M2 false-negative gate: does the sparse pipeline ever drop a pair the
exhaustive baseline flags as risky?

Two checks, because the coarse filter and the fine screen make different
kinds of promises:
  - Coarse filter (altitude bands): must be a strict superset of baseline-
    flagged pairs -- regime.py's own docstring calls this a NECESSARY
    condition, so zero misses here is a hard requirement, not a target.
  - Fine screen (relative dynamics): uses a genuinely different, stricter
    notion of risk (closing rate, not just distance) -- a baseline-flagged
    pair failing the closing-rate floor is a deliberate design choice, not
    a bug. We report it separately so it's visible, not silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from constellai.graph.filters import candidate_pairs_by_regime
from constellai.graph.screening import screen_candidate_pair
from constellai.orbital_mechanics.conjunction import ConjunctionEvent
from constellai.orbital_mechanics.tle import TLERecord
from constellai.simulation.baseline import run_baseline


@dataclass(frozen=True)
class FalseNegativeGateResult:
    baseline_flagged_count: int
    coarse_filter_missed: list[ConjunctionEvent] = field(default_factory=list)
    fine_screen_missed: list[ConjunctionEvent] = field(default_factory=list)

    @property
    def coarse_filter_false_negative_rate(self) -> float:
        if self.baseline_flagged_count == 0:
            return 0.0
        return len(self.coarse_filter_missed) / self.baseline_flagged_count

    @property
    def passed(self) -> bool:
        """The hard requirement: coarse filter must miss nothing.
        Fine-screen misses are reported but don't fail the gate -- that's
        an intentional risk-definition difference, tracked separately."""
        return len(self.coarse_filter_missed) == 0


def run_false_negative_gate(
    records: list[TLERecord],
    start: datetime,
    end: datetime,
    step: timedelta,
    threshold_km: float,
    margin_km: float,
) -> FalseNegativeGateResult:
    """Diff the exhaustive baseline against the sparse M2 pipeline."""
    baseline = run_baseline(records, start, end, step, threshold_km=threshold_km)

    candidate_ids = {
        frozenset((a.satellite_id, b.satellite_id))
        for a, b in candidate_pairs_by_regime(records, margin_km=margin_km)
    }
    records_by_id = {r.satellite_id: r for r in records}

    coarse_missed: list[ConjunctionEvent] = []
    fine_missed: list[ConjunctionEvent] = []

    for event in baseline.flagged:
        pair_key = frozenset((event.satellite_id_a, event.satellite_id_b))

        if pair_key not in candidate_ids:
            coarse_missed.append(event)
            continue

        edge = screen_candidate_pair(
            records_by_id[event.satellite_id_a],
            records_by_id[event.satellite_id_b],
            start, end, step,
            distance_threshold_km=threshold_km,
        )
        if edge is None:
            fine_missed.append(event)

    return FalseNegativeGateResult(
        baseline_flagged_count=len(baseline.flagged),
        coarse_filter_missed=coarse_missed,
        fine_screen_missed=fine_missed,
    )