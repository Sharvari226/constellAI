"""Synthetic satellite generation for controlled tests and scenarios."""

from __future__ import annotations

import math

from sgp4.api import Satrec, WGS72, jday

from constellai.common.constants import (
    EARTH_RADIUS_WGS72_KM,
    GM_EARTH_WGS72_KM3_S2,
)
from .tle import TLERecord


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
    """Build a near-circular SGP4 record at a chosen altitude."""
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
        0.0,
        0.0,
        0.0,
        0.0001,
        0.0,
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