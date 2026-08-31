"""Unit tests for build_graph() -- the combined M2 pipeline."""

from datetime import datetime, timedelta

from constellai.graph.build import build_graph
from constellai.orbital_mechanics.synthetic import make_circular_satellite


def test_build_graph_reports_correct_total_pairs():
    records = [
        make_circular_satellite(satellite_id=i, altitude_km=500.0 + i)
        for i in range(5)
    ]
    result = build_graph(
        records,
        datetime(2026, 1, 1, 0, 0, 0),
        datetime(2026, 1, 1, 0, 30, 0),
        step=timedelta(minutes=5),
    )
    assert result.total_possible_pairs == 10  # 5*4/2


def test_build_graph_coarse_candidates_never_exceed_total_pairs():
    records = [
        make_circular_satellite(satellite_id=i, altitude_km=500.0 + i)
        for i in range(5)
    ]
    result = build_graph(
        records,
        datetime(2026, 1, 1, 0, 0, 0),
        datetime(2026, 1, 1, 0, 30, 0),
        step=timedelta(minutes=5),
    )
    assert result.coarse_candidate_count <= result.total_possible_pairs


def test_build_graph_prunes_a_well_separated_shell():
    """Two satellites in a well-separated altitude shell should never
    reach the fine-screen stage at all -- this is the actual efficiency
    claim, tested end-to-end through the combined pipeline."""
    close_pair = [
        make_circular_satellite(satellite_id=1, altitude_km=500.0),
        make_circular_satellite(satellite_id=2, altitude_km=501.0),
    ]
    far_satellite = [make_circular_satellite(satellite_id=3, altitude_km=5000.0)]

    result = build_graph(
        close_pair + far_satellite,
        datetime(2026, 1, 1, 0, 0, 0),
        datetime(2026, 1, 1, 0, 30, 0),
        step=timedelta(minutes=5),
        regime_margin_km=50.0,
    )
    # 3 satellites -> 3 possible pairs, but the far satellite's altitude
    # band can't overlap either close-pair satellite -> at most 1 coarse
    # candidate (the close pair itself), not all 3.
    assert result.total_possible_pairs == 3
    assert result.coarse_candidate_count <= 1


def test_build_graph_handles_empty_and_single_satellite():
    result_empty = build_graph(
        [], datetime(2026, 1, 1), datetime(2026, 1, 1, 1), step=timedelta(minutes=5)
    )
    assert result_empty.edges == []
    assert result_empty.total_possible_pairs == 0

    single = [make_circular_satellite(satellite_id=1, altitude_km=500.0)]
    result_single = build_graph(
        single, datetime(2026, 1, 1), datetime(2026, 1, 1, 1), step=timedelta(minutes=5)
    )
    assert result_single.edges == []
    assert result_single.total_possible_pairs == 0