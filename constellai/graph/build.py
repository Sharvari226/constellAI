"""M2, tying it together: coarse filter -> fine screen -> graph.

This is the single, canonical definition of "how ConstellAI builds its
sparse dynamic graph." The false-negative gate (graph/validation.py)
depends on this being the ONLY code path that produces a graph -- if
graph construction were duplicated or reimplemented elsewhere, the gate
could pass while a real, differently-built graph still misses risks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from constellai.common.constants import (
    DEFAULT_SCREENING_DISTANCE_KM,
    DEFAULT_MIN_CLOSING_RATE_KM_S,
)
from constellai.graph.filters import candidate_pairs_by_regime
from constellai.graph.screening import GraphEdge, screen_candidate_pair
from constellai.orbital_mechanics.tle import TLERecord

# Placeholder, same status as the other threshold constants: not yet
# validated against Step-1 spec freeze. This is the margin passed to
# the coarse altitude filter specifically -- kept separate from
# DEFAULT_SCREENING_DISTANCE_KM (the fine-screen distance threshold)
# because they answer different questions: this one is "how much could
# a static altitude band be wrong by," that one is "how close counts as
# risky." Conflating them would hide that they're independent choices.
DEFAULT_REGIME_MARGIN_KM = 50.0


@dataclass(frozen=True)
class GraphBuildResult:
    """Everything build_graph() found, plus what it cost.

    Attributes
    ----------
    edges : list[GraphEdge]
        The final sparse graph -- pairs that survived both the coarse
        filter and the fine screen.
    coarse_candidate_count : int
        How many pairs survived the coarse filter (and therefore
        required the expensive fine-screen propagation). This is the
        actual efficiency number to report -- compare against
        N*(N-1)/2 to see how much the coarse filter actually saved.
    total_possible_pairs : int
        N*(N-1)/2 -- what the exhaustive baseline would have checked.
    """

    edges: list[GraphEdge]
    coarse_candidate_count: int
    total_possible_pairs: int


def build_graph(
    records: list[TLERecord],
    start: datetime,
    end: datetime,
    step: timedelta,
    regime_margin_km: float = DEFAULT_REGIME_MARGIN_KM,
    distance_threshold_km: float = DEFAULT_SCREENING_DISTANCE_KM,
    min_closing_rate_km_s: float = DEFAULT_MIN_CLOSING_RATE_KM_S,
) -> GraphBuildResult:
    """Build ConstellAI's sparse dynamic graph: coarse filter -> fine screen.

    Parameters
    ----------
    records : list[TLERecord]
    start, end, step : datetime, datetime, timedelta
        Propagation window for the fine-screen stage.
    regime_margin_km : float, optional
        Passed to the coarse altitude filter.
    distance_threshold_km, min_closing_rate_km_s : float, optional
        Passed to the fine relative-dynamics screen.

    Returns
    -------
    GraphBuildResult
    """
    n = len(records)
    total_possible_pairs = n * (n - 1) // 2 if n >= 2 else 0

    candidates = candidate_pairs_by_regime(records, margin_km=regime_margin_km)

    edges: list[GraphEdge] = []
    for record_a, record_b in candidates:
        edge = screen_candidate_pair(
            record_a, record_b, start, end, step,
            distance_threshold_km=distance_threshold_km,
            min_closing_rate_km_s=min_closing_rate_km_s,
        )
        if edge is not None:
            edges.append(edge)

    return GraphBuildResult(
        edges=edges,
        coarse_candidate_count=len(candidates),
        total_possible_pairs=total_possible_pairs,
    )