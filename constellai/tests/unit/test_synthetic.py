"""Unit tests for synthetic satellite generation."""

import pytest

from constellai.orbital_mechanics.synthetic import make_circular_satellite


def test_circular_satellite_has_near_zero_eccentricity_altitude_spread():
    record = make_circular_satellite(satellite_id=90001, altitude_km=500.0)
    altp_km = record.satrec.altp * 6378.135
    alta_km = record.satrec.alta * 6378.135

    assert altp_km == pytest.approx(500.0, abs=15.0)
    assert alta_km == pytest.approx(500.0, abs=15.0)
    assert abs(alta_km - altp_km) < 5.0


def test_different_altitudes_produce_separated_bands():
    low = make_circular_satellite(satellite_id=90001, altitude_km=500.0)
    high = make_circular_satellite(satellite_id=90002, altitude_km=1500.0)
    assert low.satrec.alta * 6378.135 < high.satrec.altp * 6378.135


def test_unique_satellite_ids_preserved():
    a = make_circular_satellite(satellite_id=90001, altitude_km=500.0)
    b = make_circular_satellite(satellite_id=90002, altitude_km=500.0)
    assert a.satellite_id != b.satellite_id
    assert a.satellite_id == "90001"


def test_propagation_succeeds_on_synthetic_satellite():
    from datetime import datetime, timedelta
    from constellai.orbital_mechanics.propagation import propagate_series

    record = make_circular_satellite(satellite_id=90001, altitude_km=500.0)
    states = propagate_series(
        record,
        datetime(2026, 1, 1, 0, 0, 0),
        datetime(2026, 1, 1, 1, 0, 0),
        timedelta(minutes=10),
    )
    # propagate_series is inclusive-start/exclusive-end: 60min/10min step
    # gives samples at 0,10,20,30,40,50 -- six states, not seven.
    assert len(states) == 6
    assert all(s.satellite_id == "90001" for s in states)