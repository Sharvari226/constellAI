"""Tests for the false-negative gate itself.

These tests check the GATE MECHANISM works correctly -- they do NOT
constitute "the gate has been run against the real project and passed."
That's a separate, deliberate next step: run run_false_negative_gate()
against a real, larger synthetic constellation and record the actual
false_negative_rate as a project result.
"""

from datetime import datetime, timedelta

from constellai.graph.validation import run_false_negative_gate
from constellai.orbital_mechanics.synthetic import make_circular_satellite


def test_gate_reports_zero_misses_when_well_separated():
    records = [
        make_circular_satellite(satellite_id=1, altitude_km=500.0),
        make_circular_satellite(satellite_id=2, altitude_km=5000.0),
        make_circular_satellite(satellite_id=3, altitude_km=10000.0),
    ]
    report = run_false_negative_gate(
        records,
        datetime(2026, 1, 1, 0, 0, 0),
        datetime(2026, 1, 1, 1, 0, 0),
        step=timedelta(minutes=5),
        distance_threshold_km=1.0,
    )
    assert report.missed_pairs == set()
    assert report.false_negative_rate == 0.0


def test_gate_correctly_detects_a_deliberately_introduced_miss():
    """A zero coarse-filter margin should prune a pair the baseline
    still flags (huge distance threshold, so the baseline flags
    regardless of true separation) -- proving the gate can actually
    catch a real discrepancy, not just report zero by construction."""
    a = make_circular_satellite(satellite_id=1, altitude_km=500.0)
    b = make_circular_satellite(satellite_id=2, altitude_km=520.0)

    report = run_false_negative_gate(
        [a, b],
        datetime(2026, 1, 1, 0, 0, 0),
        datetime(2026, 1, 1, 0, 10, 0),
        step=timedelta(minutes=2),
        distance_threshold_km=1_000_000.0,  # baseline flags regardless
        regime_margin_km=0.0,  # coarse filter should prune this pair
    )
    assert len(report.baseline_flagged_ids) == 1
    assert report.missed_pairs == report.baseline_flagged_ids
    assert report.false_negative_rate == 1.0


def test_gate_agrees_when_margin_is_generous_enough():
    """The same pair as above, but with a generous margin -- the coarse
    filter should NOT prune it this time, so nothing should be missed."""
    a = make_circular_satellite(satellite_id=1, altitude_km=500.0)
    b = make_circular_satellite(satellite_id=2, altitude_km=520.0)

    report = run_false_negative_gate(
        [a, b],
        datetime(2026, 1, 1, 0, 0, 0),
        datetime(2026, 1, 1, 0, 10, 0),
        step=timedelta(minutes=2),
        distance_threshold_km=1_000_000.0,
        regime_margin_km=50.0,  # wide enough to bridge the 20km gap
    )
    assert report.missed_pairs == set()
    assert report.false_negative_rate == 0.0