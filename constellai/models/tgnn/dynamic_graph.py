"""M2->M3 bridge: turns repeated coarse-filter + propagation into a
chronological event stream, the input format continuous-time temporal
graph models (TGN-style) actually expect -- (node_a, node_b, t, features),
ordered by time, not baked into fixed snapshots.

Candidate pairs come from M2's coarse filter (same false-negative
guarantee already validated in graph/validation.py); each candidate
pair contributes one event per observation timestep, since satellite
conjunctions don't happen at neat intervals and precision matters here
(per propagation.py's own docstring warning against reusing
fixed-interval sampling for this exact purpose -- accepted for now to
keep the bridge simple; event-driven sampling is a documented follow-up).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from constellai.graph.filters import candidate_pairs_by_regime
from constellai.orbital_mechanics.propagation import propagate_series
from constellai.orbital_mechanics.tle import TLERecord


@dataclass(frozen=True)
class GraphEvent:
    t_index: int  # position in the chronological event stream
    node_a: int
    node_b: int
    features: np.ndarray  # shape (4,): dx, dy, dz, separation_km


@dataclass(frozen=True)
class DynamicGraphData:
    node_ids: list[str]
    events: list[GraphEvent]  # chronologically sorted
    pair_labels: dict[tuple[int, int], int]  # (node_a, node_b) -> horizon label


def build_dynamic_graph(
    records: list[TLERecord],
    obs_start: datetime,
    obs_end: datetime,
    horizon_end: datetime,
    step: timedelta,
    margin_km: float,
    threshold_km: float,
) -> DynamicGraphData:
    node_ids = [r.satellite_id for r in records]
    id_to_idx = {sid: i for i, sid in enumerate(node_ids)}

    obs_states = {r.satellite_id: propagate_series(r, obs_start, obs_end, step) for r in records}
    horizon_states = {r.satellite_id: propagate_series(r, obs_end, horizon_end, step) for r in records}

    candidate_pairs = candidate_pairs_by_regime(records, margin_km=margin_km)

    events: list[GraphEvent] = []
    pair_labels: dict[tuple[int, int], int] = {}

    for a, b in candidate_pairs:
        idx_a, idx_b = id_to_idx[a.satellite_id], id_to_idx[b.satellite_id]
        states_a, states_b = obs_states[a.satellite_id], obs_states[b.satellite_id]

        for t_idx, (sa, sb) in enumerate(zip(states_a, states_b)):
            rel = sa.position_km - sb.position_km
            sep = float(np.linalg.norm(rel))
            events.append(GraphEvent(t_idx, idx_a, idx_b, np.array([*rel, sep], dtype=np.float32)))

        h_a, h_b = horizon_states[a.satellite_id], horizon_states[b.satellite_id]
        horizon_sep = np.linalg.norm(
            [x.position_km - y.position_km for x, y in zip(h_a, h_b)], axis=1
        )
        label = int(horizon_sep.min() < threshold_km) if len(horizon_sep) else 0
        pair_labels[(idx_a, idx_b)] = label

    events.sort(key=lambda e: e.t_index)  # global chronological order across all pairs

    return DynamicGraphData(node_ids=node_ids, events=events, pair_labels=pair_labels)