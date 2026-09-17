"""Builds one graph snapshot for the static-GNN baseline.

Nodes = satellites. Edges = M2's coarse-filter candidate pairs. Edge
labels use the same forecast horizon split as the LSTM baseline
(dataset.py) -- features from an observation window, label from a
later, disjoint horizon window.

Edge features now include relative speed at the last observed instant,
alongside relative position and separation -- matching dataset.py and
dynamic_graph.py so all three models see comparable features, not just
a comparable pair population.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from constellai.graph.filters import candidate_pairs_by_regime
from constellai.orbital_mechanics.propagation import propagate_series
from constellai.orbital_mechanics.tle import TLERecord


@dataclass(frozen=True)
class GraphSnapshot:
    node_ids: list[str]
    node_features: np.ndarray  # (N, 4): last position_km (3) + speed_km_s (1)
    edge_index: np.ndarray  # (E, 2) int, indices into node_ids
    edge_features: np.ndarray  # (E, 5): last-observed [dx, dy, dz, sep_km, rel_speed_km_s]
    edge_labels: np.ndarray  # (E,) int: 1 if horizon min-sep < threshold_km


def build_graph_snapshot(
    records: list[TLERecord],
    obs_start: datetime,
    obs_end: datetime,
    horizon_end: datetime,
    step: timedelta,
    margin_km: float,
    threshold_km: float,
) -> GraphSnapshot:
    node_ids = [r.satellite_id for r in records]
    id_to_idx = {sid: i for i, sid in enumerate(node_ids)}

    obs_states = {r.satellite_id: propagate_series(r, obs_start, obs_end, step) for r in records}
    horizon_states = {r.satellite_id: propagate_series(r, obs_end, horizon_end, step) for r in records}

    node_features = np.array([
        np.concatenate([
            obs_states[sid][-1].position_km,
            [np.linalg.norm(obs_states[sid][-1].velocity_km_s)],
        ])
        for sid in node_ids
    ], dtype=np.float32)

    candidate_pairs = candidate_pairs_by_regime(records, margin_km=margin_km)

    edge_index, edge_features, edge_labels = [], [], []
    for a, b in candidate_pairs:
        last_a, last_b = obs_states[a.satellite_id][-1], obs_states[b.satellite_id][-1]
        rel_pos = last_a.position_km - last_b.position_km
        sep = float(np.linalg.norm(rel_pos))
        rel_vel = last_a.velocity_km_s - last_b.velocity_km_s
        rel_speed = float(np.linalg.norm(rel_vel))

        h_a, h_b = horizon_states[a.satellite_id], horizon_states[b.satellite_id]
        horizon_sep = np.linalg.norm(
            [sa.position_km - sb.position_km for sa, sb in zip(h_a, h_b)], axis=1
        )
        label = int(horizon_sep.min() < threshold_km) if len(horizon_sep) else 0

        edge_index.append((id_to_idx[a.satellite_id], id_to_idx[b.satellite_id]))
        edge_features.append([*rel_pos, sep, rel_speed])
        edge_labels.append(label)

    return GraphSnapshot(
        node_ids=node_ids,
        node_features=node_features,
        edge_index=np.array(edge_index, dtype=np.int64).reshape(-1, 2),
        edge_features=np.array(edge_features, dtype=np.float32).reshape(-1, 5),
        edge_labels=np.array(edge_labels, dtype=np.int64),
    )