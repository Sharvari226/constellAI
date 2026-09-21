"""Real TLE catalog ingestion via CelesTrak's public API.

CelesTrak, not Space-Track, is used deliberately: it requires no
authentication (no credentials to manage, store, or leak in a research
prototype), is updated regularly, and is the source most commonly cited
in academic STM literature for public-catalog experiments. Space-Track
requires account credentials and a login flow -- out of scope for this
prototype; note this honestly if the paper's data-source claims
mention it, don't imply it's supported.

This module only PARSES what it fetches through the existing tle.py
functions -- it does not duplicate TLE-parsing logic.
"""

from __future__ import annotations

import urllib.request

from constellai.orbital_mechanics.tle import TLERecord, parse_tle_file

CELESTRAK_BASE_URL = "https://celestrak.org/NORAD/elements/gp.php"

# A few commonly-used, named CelesTrak groups -- not exhaustive; any
# valid CelesTrak GROUP name works via fetch_celestrak_group() directly.
GROUP_STARLINK = "starlink"
GROUP_ONEWEB = "oneweb"
GROUP_ACTIVE = "active"
GROUP_STATIONS = "stations"


def fetch_celestrak_group(group: str, timeout_seconds: float = 15.0) -> str:
    """Fetch raw TLE text for a named CelesTrak satellite group.

    Parameters
    ----------
    group : str
        A CelesTrak GROUP name, e.g. "starlink", "active", "stations".
        See https://celestrak.org/NORAD/elements/ for the full list.
    timeout_seconds : float, optional

    Returns
    -------
    str
        Raw TLE text (3-line format: name + two TLE lines, repeated).

    Raises
    ------
    urllib.error.URLError
        If the request fails (network issue, invalid group name, etc.)
        -- deliberately NOT caught here; callers decide how to handle
        a failed fetch (retry, fall back to synthetic data, or fail
        the experiment outright), rather than this function silently
        returning empty/partial data.
    """
    url = f"{CELESTRAK_BASE_URL}?GROUP={group}&FORMAT=tle"
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8")


def fetch_celestrak_records(group: str, timeout_seconds: float = 15.0) -> list[TLERecord]:
    """Fetch and parse a CelesTrak group in one call, reusing tle.py's
    existing 3-line parser rather than reimplementing parsing here.

    Parameters
    ----------
    group : str
    timeout_seconds : float, optional

    Returns
    -------
    list[TLERecord]

    Raises
    ------
    ValueError
        If the fetch succeeded but returned no parseable TLEs (e.g. an
        empty or malformed response) -- propagated from parse_tle_file's
        own validation, not swallowed here.
    """
    raw_text = fetch_celestrak_group(group, timeout_seconds=timeout_seconds)

    # parse_tle_file expects a file path, not raw text -- write to a
    # temp file rather than duplicating its parsing logic for a string
    # input. This keeps ONE canonical TLE-parsing code path in the
    # whole project (tle.py), matching the project's established
    # convention (one canonical graph builder, one canonical constants
    # module, etc.).
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tle", delete=False) as f:
        f.write(raw_text)
        temp_path = f.name

    try:
        return parse_tle_file(temp_path)
    finally:
        import os
        os.unlink(temp_path)