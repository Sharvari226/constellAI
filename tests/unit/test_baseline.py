"""Unit tests for the Step-3 non-ML conjunction baseline.

Uses three synthetic satellites derived from the Vallado reference TLE
(same physical orbit, mean anomaly offset by ~90 deg increments) so each
has a genuinely different, propagatable position -- not just relabeled
copies of the same state. Checksums are deliberately not recomputed:
python-sgp4 does not validate the TLE checksum digit, so the trailing
digit being "wrong" has no effect on propagation.
"""

from datetime import datetime, timedelta

import pytest

from constellai.orbital_mechanics.tle import parse_tle
from constellai.simulation.baseline import run_baseline

LINE1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
_LINE2_BASE = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"


def _line2_variant(catalog_number: str, mean_anomaly: str) -> str:
    """Swap the catalog-number (0-indexed [2:7]) and mean-anomaly
    (0-indexed [43:51]) fields for a synthetic variant satellite.

    Both must be varied together: catalog number is what run_baseline
    uses as satellite_id to key its propagation cache -- reusing the
    same catalog number across "different" satellites silently
    collapses them into one cached state series. Field widths must
    stay exact (5 and 8 chars) or every later column shifts.
    """
    assert len(catalog_number) == 5, "catalog-number field must be exactly 5 chars wide"
    assert len(mean_anomaly) == 8, "mean-anomaly field must be exactly 8 chars wide"
    line2 = _LINE2_BASE[:2] + catalog_number + _LINE2_BASE[7:43] + mean_anomaly + _LINE2_BASE[51:]
    assert len(line2) == len(_LINE2_BASE)
    return line2


@pytest.fixture
def three_satellites():
    return [
        parse_tle(LINE1, _line2_variant("00005", " 19.3264"), name="Sat-A"),
        parse_tle(LINE1, _line2_variant("00006", "109.3264"), name="Sat-B"),
        parse_tle(LINE1, _line2_variant("00007", "199.3264"), name="Sat-C"),
    ]


def test_baseline_screens_every_pair(three_satellites):
    start = datetime(2000, 6, 29, 12, 0, 0)
    result = run_baseline(
        three_satellites, start, start + timedelta(minutes=10),
        step=timedelta(minutes=5),
        threshold_km=1.0,
    )
    assert result.pairs_screened == 3


def test_baseline_flags_nothing_below_a_tight_threshold(three_satellites):
    start = datetime(2000, 6, 29, 12, 0, 0)
    result = run_baseline(
        three_satellites, start, start + timedelta(minutes=10),
        step=timedelta(minutes=5),
        threshold_km=1.0,
    )
    assert result.flagged == []


def test_baseline_flags_everything_below_a_huge_threshold(three_satellites):
    start = datetime(2000, 6, 29, 12, 0, 0)
    result = run_baseline(
        three_satellites, start, start + timedelta(minutes=10),
        step=timedelta(minutes=5),
        threshold_km=1_000_000.0,
    )
    assert len(result.flagged) == result.pairs_screened == 3


def test_baseline_rejects_fewer_than_two_satellites(three_satellites):
    start = datetime(2000, 6, 29, 12, 0, 0)
    with pytest.raises(ValueError):
        run_baseline(
            three_satellites[:1], start, start + timedelta(minutes=10),
            step=timedelta(minutes=5),
        )


def test_baseline_threshold_is_recorded_on_the_result(three_satellites):
    start = datetime(2000, 6, 29, 12, 0, 0)
    result = run_baseline(
        three_satellites, start, start + timedelta(minutes=10),
        step=timedelta(minutes=5),
        threshold_km=42.0,
    )
    assert result.threshold_km == 42.0