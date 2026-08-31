"""Unit tests for orbital regime (altitude band) computation and overlap."""

import pytest

from constellai.orbital_mechanics.regime import altitude_band, bands_overlap
from constellai.orbital_mechanics.synthetic import make_circular_satellite


def test_altitude_band_matches_target_altitude():
    record = make_circular_satellite(satellite_id=90001, altitude_km=500.0)
    band = altitude_band(record)
    assert band.perigee_alt_km == pytest.approx(500.0, abs=15.0)
    assert band.apogee_alt_km == pytest.approx(500.0, abs=15.0)


def test_widely_separated_bands_do_not_overlap():
    low = altitude_band(make_circular_satellite(satellite_id=90001, altitude_km=500.0))
    high = altitude_band(make_circular_satellite(satellite_id=90002, altitude_km=1500.0))
    assert not bands_overlap(low, high, margin_km=10.0)


def test_same_altitude_bands_overlap():
    a = altitude_band(make_circular_satellite(satellite_id=90001, altitude_km=500.0))
    b = altitude_band(make_circular_satellite(satellite_id=90002, altitude_km=500.0))
    assert bands_overlap(a, b, margin_km=10.0)


def test_margin_can_bridge_a_near_miss():
    low = altitude_band(make_circular_satellite(satellite_id=90001, altitude_km=500.0))
    high = altitude_band(make_circular_satellite(satellite_id=90002, altitude_km=550.0))
    assert not bands_overlap(low, high, margin_km=1.0)
    assert bands_overlap(low, high, margin_km=30.0)