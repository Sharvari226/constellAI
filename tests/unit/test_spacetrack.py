"""Unit tests for Space-Track integration.

Split deliberately into two groups: credential-handling tests (run
always, no network needed) and actual network tests (skipped
automatically if credentials aren't set -- so this test file doesn't
break for anyone who hasn't registered a Space-Track account, unlike
CelesTrak's tests which need no auth at all).
"""

import os

import pytest

from constellai.orbital_mechanics.spacetrack import (
    SpaceTrackAuthError,
    _get_credentials,
    fetch_spacetrack_records,
)

HAS_CREDENTIALS = bool(os.environ.get("SPACETRACK_USERNAME")) and bool(os.environ.get("SPACETRACK_PASSWORD"))


def test_missing_credentials_raises_clear_error(monkeypatch):
    monkeypatch.delenv("SPACETRACK_USERNAME", raising=False)
    monkeypatch.delenv("SPACETRACK_PASSWORD", raising=False)
    with pytest.raises(SpaceTrackAuthError, match="SPACETRACK_USERNAME"):
        _get_credentials()


def test_partial_credentials_still_raises(monkeypatch):
    monkeypatch.setenv("SPACETRACK_USERNAME", "someuser")
    monkeypatch.delenv("SPACETRACK_PASSWORD", raising=False)
    with pytest.raises(SpaceTrackAuthError):
        _get_credentials()


def test_complete_credentials_do_not_raise(monkeypatch):
    monkeypatch.setenv("SPACETRACK_USERNAME", "someuser")
    monkeypatch.setenv("SPACETRACK_PASSWORD", "somepass")
    username, password = _get_credentials()
    assert username == "someuser"
    assert password == "somepass"


@pytest.mark.network
@pytest.mark.skipif(not HAS_CREDENTIALS, reason="SPACETRACK_USERNAME/PASSWORD not set")
def test_fetch_real_iss_tle():
    """ISS NORAD ID 25544 -- always in the catalog, good smoke test."""
    records = fetch_spacetrack_records(norad_ids=[25544])
    assert len(records) >= 1
    assert records[0].satellite_id == "25544"