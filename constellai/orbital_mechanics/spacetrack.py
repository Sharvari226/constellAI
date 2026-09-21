"""Real TLE ingestion via Space-Track -- historical multi-epoch queries,
unlike CelesTrak's public "current catalog only" feed.

Requires a free Space-Track account (space-track.org). Credentials are
read ONLY from environment variables (SPACETRACK_USERNAME,
SPACETRACK_PASSWORD) -- never hardcoded, never read from a file that
could end up committed to the repo. This is a real security discipline
choice for a project with a public GitHub repo, not boilerplate.

Session-based auth: POST credentials once -> receive a session cookie
-> reuse that cookie for subsequent queries, standard Space-Track API
pattern (different from CelesTrak's bare, unauthenticated GET).
"""

from __future__ import annotations

import os
import tempfile
import urllib.error
import urllib.request

from constellai.orbital_mechanics.tle import TLERecord, parse_tle_file

SPACETRACK_BASE_URL = "https://www.space-track.org"
LOGIN_URL = f"{SPACETRACK_BASE_URL}/ajaxauth/login"


class SpaceTrackAuthError(RuntimeError):
    """Raised when credentials are missing or login fails -- distinct
    from a generic network error, so callers can give a clear,
    actionable message rather than a raw HTTP failure."""


def _get_credentials() -> tuple[str, str]:
    username = os.environ.get("SPACETRACK_USERNAME")
    password = os.environ.get("SPACETRACK_PASSWORD")
    if not username or not password:
        raise SpaceTrackAuthError(
            "SPACETRACK_USERNAME and SPACETRACK_PASSWORD environment "
            "variables must both be set. Register a free account at "
            "https://www.space-track.org, then set these in your shell "
            "(never commit them to a file)."
        )
    return username, password


def _login(opener: urllib.request.OpenerDirector) -> None:
    """Authenticates and stores the session cookie in the given opener's
    cookie jar (via urllib's cookie-handling machinery) for reuse."""
    username, password = _get_credentials()
    data = f"identity={username}&password={password}".encode("utf-8")
    request = urllib.request.Request(LOGIN_URL, data=data, method="POST")
    try:
        opener.open(request, timeout=15.0)
    except urllib.error.HTTPError as e:
        raise SpaceTrackAuthError(f"Space-Track login failed (HTTP {e.code}) -- check credentials.") from e


def fetch_spacetrack_tles(
    norad_ids: list[int],
    epoch_start: str | None = None,
    epoch_end: str | None = None,
) -> str:
    """Fetch raw TLE text for specific NORAD catalog IDs, optionally
    within a historical epoch range -- the actual capability CelesTrak's
    public feed doesn't offer (current catalog only).

    Parameters
    ----------
    norad_ids : list[int]
        NORAD catalog numbers to fetch.
    epoch_start, epoch_end : str, optional
        Date strings in "YYYY-MM-DD" format. If both given, queries
        that historical epoch range; if omitted, returns latest.

    Returns
    -------
    str
        Raw TLE text.

    Raises
    ------
    SpaceTrackAuthError
        If credentials are missing or login fails.
    urllib.error.URLError
        If the query request itself fails, after successful login --
        propagated, not swallowed, same convention as fetch.py.
    """
    cookie_jar = urllib.request.HTTPCookieProcessor()
    opener = urllib.request.build_opener(cookie_jar)
    _login(opener)

    ids_str = ",".join(str(i) for i in norad_ids)
    if epoch_start and epoch_end:
        query = (
            f"{SPACETRACK_BASE_URL}/basicspacedata/query/class/tle/"
            f"NORAD_CAT_ID/{ids_str}/EPOCH/{epoch_start}--{epoch_end}/"
            f"orderby/EPOCH asc/format/tle"
        )
    else:
        query = (
            f"{SPACETRACK_BASE_URL}/basicspacedata/query/class/tle_latest/"
            f"NORAD_CAT_ID/{ids_str}/ORDINAL/1/format/tle"
        )

    request = urllib.request.Request(query)
    with opener.open(request, timeout=15.0) as response:
        return response.read().decode("utf-8")


def fetch_spacetrack_records(
    norad_ids: list[int],
    epoch_start: str | None = None,
    epoch_end: str | None = None,
) -> list[TLERecord]:
    """Fetch and parse in one call, reusing tle.py's existing parser --
    same one-canonical-parsing-path convention as fetch.py (CelesTrak).

    Raises
    ------
    SpaceTrackAuthError, urllib.error.URLError
        Propagated from fetch_spacetrack_tles.
    ValueError
        If the fetch succeeded but returned no parseable TLEs.
    """
    raw_text = fetch_spacetrack_tles(norad_ids, epoch_start, epoch_end)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tle", delete=False) as f:
        f.write(raw_text)
        temp_path = f.name

    try:
        return parse_tle_file(temp_path)
    finally:
        os.unlink(temp_path)