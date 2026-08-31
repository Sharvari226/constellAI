"""Coarse candidate filtering by orbital altitude regime."""

from __future__ import annotations

from constellai.orbital_mechanics.regime import altitude_band, bands_overlap
from constellai.orbital_mechanics.tle import TLERecord


def candidate_pairs_by_regime(
    records: list[TLERecord],
    margin_km: float,
) -> list[tuple[TLERecord, TLERecord]]:
    """Return record pairs whose altitude bands overlap."""
    if len(records) < 2:
        return []
    bands = {record.satellite_id: altitude_band(record) for record in records}
    ordered = sorted(records, key=lambda record: bands[record.satellite_id].perigee_alt_km)
    candidates: list[tuple[TLERecord, TLERecord]] = []
    active: list[TLERecord] = []
    for record in ordered:
        band = bands[record.satellite_id]
        active = [
            other for other in active
            if bands[other.satellite_id].apogee_alt_km + margin_km
            >= band.perigee_alt_km - margin_km
        ]
        for other in active:
            if bands_overlap(bands[other.satellite_id], band, margin_km):
                candidates.append((other, record))
        active.append(record)
    return candidates