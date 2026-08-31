"""The false-negative gate: does the sparse graph miss real risks that
exhaustive pairwise checking would catch?

This is the single most important validation in the M2 layer. The
entire "O(N^2) -> O(|E_t|)" efficiency claim this project rests on is
worthless if the sparse graph silently drops genuine conjunction risks
to get there. This module runs build_graph() and run_baseline() on the
identical scenario and reports exactly what, if anything, the sparse
graph missed -- as a number, not an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from constellai.common.constants import DEFAULT_SCREENING_DISTANCE_KM
from constellai.graph.build import DEFAULT_REGIME_MARGIN_KM, GraphBuildResult, build_graph
from constellai.orbital_mechanics.tle import TLERecord
from constellai.simulation.baseline import BaselineResult, run_baseline


@dataclass(frozen=True)
class FalseNegativeReport:
    """Comparison between the sparse graph and the exhaustive baseline."""

    baseline_flagged_ids: set[frozenset]
    graph_flagged_ids: set[frozenset]
    missed_pairs: set[frozenset]
    false_negative_rate: float


def run_false_negative_gate(
    records: list[TLERecord],
    start: datetime,
    end: datetime,
    step: timedelta,
    distance_threshold_km: float = DEFAULT_SCREENING_DISTANCE_KM,
    regime_margin_km: float = DEFAULT_REGIME_MARGIN_KM,
) -> FalseNegativeReport:
    """Run both the exhaustive baseline and the sparse graph on the same
    scenario, and report what the graph missed.

    Parameters
    ----------
    records : list[TLERecord]
    start, end, step : datetime, datetime, timedelta
        Identical propagation window used for both methods.
    distance_threshold_km : float, optional
        Passed to BOTH methods identically -- an unequal threshold
        would make any difference in flagged pairs uninterpretable.
    regime_margin_km : float, optional
        Passed to build_graph()'s coarse filter. Exposed here
        explicitly (not hidden behind build_graph's default) so this
        gate can actually be exercised at different filter tightness --
        the whole point of a validation gate is being able to probe it,
        not just call it once with fixed settings.

    Returns
    -------
    FalseNegativeReport
    """
    baseline_result: BaselineResult = run_baseline(
        records, start, end, step, threshold_km=distance_threshold_km,
    )
    graph_result: GraphBuildResult = build_graph(
        records, start, end, step,
        regime_margin_km=regime_margin_km,
        distance_threshold_km=distance_threshold_km,
    )

    baseline_ids = {
        frozenset((e.satellite_id_a, e.satellite_id_b))
        for e in baseline_result.flagged
    }
    graph_ids = {
        frozenset(edge.satellite_ids) for edge in graph_result.edges
    }

    missed = baseline_ids - graph_ids
    rate = len(missed) / len(baseline_ids) if baseline_ids else 0.0

    return FalseNegativeReport(
        baseline_flagged_ids=baseline_ids,
        graph_flagged_ids=graph_ids,
        missed_pairs=missed,
        false_negative_rate=rate,
    )