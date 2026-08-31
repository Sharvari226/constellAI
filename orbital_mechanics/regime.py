"""Orbital regime characterization: perigee/apogee altitude bands.

Physical basis for M2's coarse filter (graph/filters.py): two satellites
can only conjunct if their altitude ranges overlap -- a NECESSARY, not
sufficient, condition. Non-overlapping bands guarantee no risk;
overlapping bands don't guarantee risk (inclination/RAAN differences
can keep same-altitude orbits from ever crossing). That asymmetry is
what a coarse pre-filter needs: it must never produce false negatives,
and it's fine to let false positives through to the expensive stage.

Perigee/apogee altitude come directly from SGP4's own initialization
(satrec.altp, satrec.alta) rather than re-derived here.
"""

from __future__ import annotations

from dataclasses import dataclass

from constellai.common.constants import EARTH_RADIUS_WGS72_KM
from constellai.orbital_mechanics.tle import TLERecord


@dataclass(frozen=True)
class AltitudeBand:
    """A satellite's perigee-to-apogee altitude range, in km."""

    perigee_alt_km: float
    apogee_alt_km: float


def altitude_band(record: TLERecord) -> AltitudeBand:
    """Compute a satellite's altitude band from its parsed TLE."""
    return AltitudeBand(
        perigee_alt_km=record.satrec.altp * EARTH_RADIUS_WGS72_KM,
        apogee_alt_km=record.satrec.alta * EARTH_RADIUS_WGS72_KM,
    )


def bands_overlap(a: AltitudeBand, b: AltitudeBand, margin_km: float) -> bool:
    """Whether two altitude bands overlap, inflated by a safety margin.

    margin_km must cover both the conjunction screening distance and how
    much a "static" band can drift within the screening window from
    drag/J2 -- not yet validated against either; treat as provisional.
    """
    a_lo, a_hi = a.perigee_alt_km - margin_km, a.apogee_alt_km + margin_km
    b_lo, b_hi = b.perigee_alt_km - margin_km, b.apogee_alt_km + margin_km
    return not (a_hi < b_lo or b_hi < a_lo)