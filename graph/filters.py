"""M2 coarse filter: reduce N*(N-1)/2 candidate pairs using altitude bands.

Honest complexity note: this sweep is only cheap when altitude bands are
spread out. For a single-shell mega-constellation (many satellites at
nearly the same altitude -- the realistic worst case, not hypothetical),
this degrades toward the same O(N^2) the baseline already has. Further
reduction within a shared shell has to come from something finer than
altitude alone (true relative-dynamics filtering -- M3's job).
"""

from __future__ import annotations

from constellai.orbital_mechanics.regime import altitude_band, bands_overlap
from constellai.orbital_mechanics.tle import TLERecord


def candidate_pairs_by_regime(
    records: list[TLERecord],
    margin_km: float,
) -> list[tuple[TLERecord, TLERecord]]:
    """Return satellite pairs whose altitude bands overlap.

    This is a superset of true conjunction risks by design -- callers
    still run the expensive relative-dynamics screen on survivors; this
    function only narrows the field.
    """
    if len(records) < 2:
        return []

    bands = {r.satellite_id: altitude_band(r) for r in records}
    ordered = sorted(records, key=lambda r: bands[r.satellite_id].perigee_alt_km)

    candidates: list[tuple[TLERecord, TLERecord]] = []
    active: list[TLERecord] = []

    for record in ordered:
        band = bands[record.satellite_id]

        active = [
            a for a in active
            if bands[a.satellite_id].apogee_alt_km + margin_km
            >= band.perigee_alt_km - margin_km
        ]

        for other in active:
            if bands_overlap(bands[other.satellite_id], band, margin_km):
                candidates.append((other, record))

        active.append(record)

    return candidates