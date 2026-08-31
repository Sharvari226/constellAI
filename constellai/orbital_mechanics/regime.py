"""Orbital regime characterization using perigee/apogee altitude bands."""

from __future__ import annotations

from dataclasses import dataclass

from constellai.common.constants import EARTH_RADIUS_WGS72_KM
from .tle import TLERecord


@dataclass(frozen=True)
class AltitudeBand:
    """A satellite's perigee-to-apogee altitude range, in km."""

    perigee_alt_km: float
    apogee_alt_km: float


def altitude_band(record: TLERecord) -> AltitudeBand:
    """Return the SGP4-derived perigee and apogee altitudes."""
    return AltitudeBand(
        perigee_alt_km=record.satrec.altp * EARTH_RADIUS_WGS72_KM,
        apogee_alt_km=record.satrec.alta * EARTH_RADIUS_WGS72_KM,
    )


def bands_overlap(a: AltitudeBand, b: AltitudeBand, margin_km: float) -> bool:
    """Return whether two altitude bands overlap after adding a margin."""
    a_low, a_high = a.perigee_alt_km - margin_km, a.apogee_alt_km + margin_km
    b_low, b_high = b.perigee_alt_km - margin_km, b.apogee_alt_km + margin_km
    return not (a_high < b_low or b_high < a_low)