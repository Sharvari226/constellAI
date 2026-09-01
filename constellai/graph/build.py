"""M2 canonical pipeline: coarse filter -> fine screen, in one call.

This is the single entry point downstream code (M3, M6, scripts) should
use instead of calling filters.py and screening.py separately -- keeps
the two-stage pipeline's actual sequence in one place, matching the
architecture diagram's Stage A -> Stage B flow exactly.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from constellai.graph.filters import candidate_pairs_by_regime
from constellai.graph.screening import GraphEdge, screen_candidate_pair
from constellai.orbital_mechanics.tle import TLERecord


def build_graph(
    records: list[TLERecord],
    start: datetime,
    end: datetime,
    step: timedelta,
    margin_km: float,
    distance_threshold_km: float,
    min_closing_rate_km_s: float = 0.0,
) -> list[GraphEdge]:
    """Coarse-filter candidates, then fine-screen survivors into edges."""
    candidates = candidate_pairs_by_regime(records, margin_km=margin_km)

    edges: list[GraphEdge] = []
    for a, b in candidates:
        edge = screen_candidate_pair(
            a, b, start, end, step,
            distance_threshold_km=distance_threshold_km,
            min_closing_rate_km_s=min_closing_rate_km_s,
        )
        if edge is not None:
            edges.append(edge)

    return edges