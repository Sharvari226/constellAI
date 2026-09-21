"""Unit tests for the combined M2 pipeline (build_graph)."""

from datetime import datetime, timedelta

import numpy as np

from constellai.graph.build import build_graph
from constellai.models.tgnn.dynamic_graph import build_dynamic_graph
from constellai.orbital_mechanics.propagation import propagate_series
from constellai.orbital_mechanics.synthetic import make_circular_satellite

START = datetime(2026, 1, 1)
END = START + timedelta(hours=1)
STEP = timedelta(minutes=5)


def test_build_graph_returns_edges_for_close_satellites():
    a = make_circular_satellite(satellite_id=1, altitude_km=500.0)
    b = make_circular_satellite(satellite_id=2, altitude_km=500.0)

    edges = build_graph(
        [a, b], START, END, STEP,
        margin_km=10.0, distance_threshold_km=15000.0, min_closing_rate_km_s=0.0,
    )
    assert len(edges) == 1
    assert edges[0].satellite_ids == ("1", "2")


def test_build_graph_excludes_widely_separated_altitudes():
    a = make_circular_satellite(satellite_id=1, altitude_km=500.0)
    b = make_circular_satellite(satellite_id=2, altitude_km=5000.0)

    edges = build_graph([a, b], START, END, STEP, margin_km=10.0, distance_threshold_km=15000.0)
    assert edges == []


def test_build_graph_handles_no_candidates():
    single = [make_circular_satellite(satellite_id=1, altitude_km=500.0)]
    edges = build_graph(single, START, END, STEP, margin_km=10.0, distance_threshold_km=100.0)
    assert edges == []


def test_build_dynamic_graph_includes_relative_speed_feature():
    a = make_circular_satellite(satellite_id=1, altitude_km=500.0)
    b = make_circular_satellite(satellite_id=2, altitude_km=500.0)

    graph = build_dynamic_graph(
        [a, b], START, END, START + timedelta(hours=2), STEP,
        margin_km=10.0, threshold_km=300.0,
    )

    assert graph.events
    assert all(event.features.shape == (5,) for event in graph.events)

    states_a = propagate_series(a, START, END, STEP)
    states_b = propagate_series(b, START, END, STEP)
    relative_speed = np.linalg.norm(states_a[0].velocity_km_s - states_b[0].velocity_km_s)
    assert np.isclose(graph.events[0].features[-1], relative_speed)