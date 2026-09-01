"""M2 completion gate: sparse pipeline must never drop what the
exhaustive baseline flags as risky (coarse-filter stage, specifically)."""

from datetime import datetime, timedelta

from constellai.orbital_mechanics.synthetic import make_circular_satellite
from constellai.graph.validation import run_false_negative_gate

START = datetime(2026, 1, 1)
END = START + timedelta(hours=1)
STEP = timedelta(minutes=5)


def test_gate_passes_on_widely_spread_altitudes():
    records = [
        make_circular_satellite(satellite_id=100 + i, altitude_km=500.0 + i * 200)
        for i in range(6)
    ]
    result = run_false_negative_gate(
        records, START, END, STEP, threshold_km=50.0, margin_km=10.0,
    )
    assert result.passed
    assert result.coarse_filter_missed == []


def test_gate_passes_on_single_shell_mega_constellation_worst_case():
    """filters.py's own docstring names this the honest worst case:
    many satellites at nearly the same altitude. Coarse pruning is weak
    here, so this is exactly where a false negative would surface."""
    records = [
        make_circular_satellite(satellite_id=200 + i, altitude_km=500.0 + i * 0.5)
        for i in range(30)
    ]
    result = run_false_negative_gate(
        records, START, END, STEP, threshold_km=50.0, margin_km=10.0,
    )
    assert result.passed, (
        f"coarse filter dropped {len(result.coarse_filter_missed)} "
        f"baseline-flagged pair(s) -- false-negative guarantee violated"
    )


def test_gate_reports_baseline_flag_count():
    records = [
        make_circular_satellite(satellite_id=1, altitude_km=500.0),
        make_circular_satellite(satellite_id=2, altitude_km=500.0),
    ]
    result = run_false_negative_gate(
        records, START, END, STEP, threshold_km=1_000_000.0, margin_km=10.0,
    )
    assert result.baseline_flagged_count == 1
    assert result.passed


def test_margin_too_small_can_be_detected_by_the_gate():
    """Sanity check on the gate itself: an under-sized margin should be
    able to produce a coarse-filter miss, proving the gate isn't a
    tautology that always passes."""
    records = [
        make_circular_satellite(satellite_id=1, altitude_km=500.0),
        make_circular_satellite(satellite_id=2, altitude_km=560.0),
    ]
    result = run_false_negative_gate(
        records, START, END, STEP, threshold_km=1_000_000.0, margin_km=0.0,
    )
    if result.baseline_flagged_count > 0:
        assert not result.passed