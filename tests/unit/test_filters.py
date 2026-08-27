"""Unit tests for the M2 coarse filter (sweep-line candidate pairing)."""

from constellai.graph.filters import candidate_pairs_by_regime
from constellai.orbital_mechanics.synthetic import make_circular_satellite


def _ids(pairs):
    return {frozenset((a.satellite_id, b.satellite_id)) for a, b in pairs}


def test_three_altitude_clusters_only_pair_within_cluster():
    records = [
        make_circular_satellite(satellite_id=1, altitude_km=500.0),
        make_circular_satellite(satellite_id=2, altitude_km=505.0),
        make_circular_satellite(satellite_id=3, altitude_km=1500.0),
        make_circular_satellite(satellite_id=4, altitude_km=1505.0),
    ]
    pairs = candidate_pairs_by_regime(records, margin_km=10.0)
    found = _ids(pairs)

    assert frozenset(("1", "2")) in found
    assert frozenset(("3", "4")) in found
    for cross in [("1", "3"), ("1", "4"), ("2", "3"), ("2", "4")]:
        assert frozenset(cross) not in found


def test_pruning_actually_reduces_pair_count_vs_exhaustive():
    low_shell = [
        make_circular_satellite(satellite_id=100 + i, altitude_km=500.0 + i)
        for i in range(5)
    ]
    high_shell = [
        make_circular_satellite(satellite_id=200 + i, altitude_km=2000.0 + i)
        for i in range(5)
    ]
    records = low_shell + high_shell

    exhaustive_pair_count = len(records) * (len(records) - 1) // 2
    pairs = candidate_pairs_by_regime(records, margin_km=10.0)

    assert exhaustive_pair_count == 45
    assert len(pairs) < exhaustive_pair_count
    assert len(pairs) <= 20


def test_single_satellite_or_empty_returns_no_pairs():
    assert candidate_pairs_by_regime([], margin_km=10.0) == []
    single = [make_circular_satellite(satellite_id=1, altitude_km=500.0)]
    assert candidate_pairs_by_regime(single, margin_km=10.0) == []


def test_zero_margin_still_pairs_identical_altitudes():
    records = [
        make_circular_satellite(satellite_id=1, altitude_km=500.0),
        make_circular_satellite(satellite_id=2, altitude_km=500.0),
    ]
    pairs = candidate_pairs_by_regime(records, margin_km=0.0)
    assert len(pairs) == 1