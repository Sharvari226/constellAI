"""Unit tests for real TLE fetching.

These tests make ACTUAL network calls to CelesTrak -- deliberately, not
mocked, since the entire point is confirming this project can genuinely
ingest real, current catalog data, not just that a mock returns what we
told it to. Marked so they can be skipped in offline/CI environments
that don't have network access, once M7's CI setup exists.
"""

import pytest

from constellai.orbital_mechanics.fetch import (
    GROUP_STATIONS,
    fetch_celestrak_group,
    fetch_celestrak_records,
)


@pytest.mark.network
def test_fetch_celestrak_group_returns_nonempty_text():
    raw = fetch_celestrak_group(GROUP_STATIONS)
    assert len(raw) > 0
    assert "ISS" in raw.upper() or "ZARYA" in raw.upper()  # the ISS is always in the stations group


@pytest.mark.network
def test_fetch_celestrak_records_returns_parsed_tles():
    records = fetch_celestrak_records(GROUP_STATIONS)
    assert len(records) > 0
    assert all(r.satellite_id.isdigit() for r in records)


@pytest.mark.network
def test_fetched_records_are_propagatable():
    """The real end-to-end claim: a satellite fetched from CelesTrak
    RIGHT NOW must work with the existing, already-validated
    propagation pipeline without any special-casing."""
    from datetime import datetime, timedelta
    from constellai.orbital_mechanics.propagation import propagate_series

    records = fetch_celestrak_records(GROUP_STATIONS)
    record = records[0]

    states = propagate_series(
        record, datetime.utcnow(), datetime.utcnow() + timedelta(hours=1), timedelta(minutes=10)
    )
    assert len(states) > 0