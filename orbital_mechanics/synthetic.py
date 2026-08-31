"""Synthetic satellite generation for controlled tests and scenarios.

Constructs TLERecords directly via SGP4's sgp4init(), bypassing TLE
string formatting entirely. Useful for two things: controlled test
fixtures here (exact altitude control, instead of hand-editing TLE text
and hoping it lands right), and later, generating synthetic
constellations of arbitrary size N for the false-negative gate and
scalability experiments.
"""

from __future__ import annotations

import math

from sgp4.api import Satrec, WGS72, jday

from constellai.common.constants import GM_EARTH_WGS72_KM3_S2, EARTH_RADIUS_WGS72_KM
from constellai.orbital_mechanics.tle import TLERecord


def make_circular_satellite(
    satellite_id: int,
    altitude_km: float,
    inclination_rad: float = 0.0,
    raan_rad: float = 0.0,
    mean_anomaly_rad: float = 0.0,
    epoch_year: int = 2026,
    epoch_month: int = 1,
    epoch_day: int = 1,
) -> TLERecord:
    """Build a near-circular synthetic satellite at a chosen altitude.

    Eccentricity is fixed at a small nonzero value (0.0001), not exactly
    zero -- SGP4's handling of exactly-circular orbits (argument of
    perigee undefined) is a known edge case not worth introducing here.
    Note: altp/alta recovered from this construction land within ~5-15 km
    of the target altitude, not exactly on it -- SGP4 applies its own
    internal mean-to-osculating-element correction distinct from this
    function's naive vis-viva semi-major-axis calculation. That's fine
    for a coarse altitude-band filter; it would not be fine for anything
    needing precise orbit shape.
    """
    semi_major_axis_km = EARTH_RADIUS_WGS72_KM + altitude_km
    mean_motion_rad_per_min = (
        math.sqrt(GM_EARTH_WGS72_KM3_S2 / semi_major_axis_km**3) * 60.0
    )

    satrec = Satrec()
    jd, fr = jday(epoch_year, epoch_month, epoch_day, 0, 0, 0)
    epoch_days_since_1949dec31 = (jd - 2433281.5) + fr

    satrec.sgp4init(
        WGS72,
        "i",
        satellite_id,
        epoch_days_since_1949dec31,
        0.0,  # bstar
        0.0,  # ndot (deprecated, unused)
        0.0,  # nddot (deprecated, unused)
        0.0001,  # eccentricity
        0.0,  # argument of perigee, rad
        inclination_rad,
        mean_anomaly_rad,
        mean_motion_rad_per_min,
        raan_rad,
    )
    if satrec.error != 0:
        raise ValueError(f"sgp4init failed with error code {satrec.error}")

    return TLERecord(
        satellite_id=str(satellite_id),
        name=f"SYNTH-{satellite_id}",
        line1="",
        line2="",
        satrec=satrec,
    )