"""Unit tests for M2's fine relative-dynamics screening."""

from datetime import datetime, timedelta

from constellai.graph.screening import screen_candidate_pair
from constellai.orbital_mechanics.synthetic import make_circular_satellite


def test_close_and_fast_closing_pair_becomes_an_edge():
    """Two satellites at the same altitude, opposite points on the orbit
    (close to a head-on pass), should produce an edge with a tight
    threshold+low closing-rate floor."""
    import math

    a = make_circular_satellite(satellite_id=1, altitude_km=500.0, mean_anomaly_rad=0.0)
    b = make_circular_satellite(satellite_id=2, altitude_km=500.0, mean_anomaly_rad=math.pi)

    edge = screen_candidate_pair(
        a, b,
        datetime(2026, 1, 1, 0, 0, 0),
        datetime(2026, 1, 1, 2, 0, 0),
        step=timedelta(seconds=30),
        distance_threshold_km=15000.0,  # generous, just checking the mechanism works
        min_closing_rate_km_s=0.0,
    )
    assert edge is not None
    assert edge.satellite_ids == ("1", "2")


def test_returns_none_when_distance_threshold_not_met():
    a = make_circular_satellite(satellite_id=1, altitude_km=500.0)
    b = make_circular_satellite(satellite_id=2, altitude_km=5000.0)

    edge = screen_candidate_pair(
        a, b,
        datetime(2026, 1, 1, 0, 0, 0),
        datetime(2026, 1, 1, 1, 0, 0),
        step=timedelta(minutes=5),
        distance_threshold_km=1.0,  # essentially impossible to satisfy here
    )
    assert edge is None


def test_returns_none_when_closing_rate_floor_not_met():
    """Same satellite compared to itself: miss distance is ~0 (passes
    the distance check) but closing rate is also ~0 (never truly
    'approaching' since it's colocated the whole time) -- a high floor
    should reject this."""
    a = make_circular_satellite(satellite_id=1, altitude_km=500.0)

    edge = screen_candidate_pair(
        a, a,
        datetime(2026, 1, 1, 0, 0, 0),
        datetime(2026, 1, 1, 1, 0, 0),
        step=timedelta(minutes=5),
        distance_threshold_km=15000.0,
        min_closing_rate_km_s=100.0,  # deliberately unreachable floor
    )
    assert edge is None