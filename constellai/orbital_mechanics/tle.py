"""TLE (Two-Line Element) parsing.

This module does exactly one job: turn TLE text into a TLERecord. No
propagation, no physics happens here -- that's orbital_mechanics/
propagation.py, deliberately kept separate.

Actual line parsing is delegated to sgp4's own Satrec.twoline2rv --
the maintained, standard implementation. We wrap its result in our own
TLERecord so the rest of the codebase depends on our interface, not
directly on the third-party object's shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from sgp4.api import Satrec, WGS72


@dataclass(frozen=True)
class TLERecord:
    """A parsed satellite orbital record."""

    satellite_id: str
    name: str
    line1: str
    line2: str
    satrec: Satrec


def parse_tle(line1: str, line2: str, name: str | None = None) -> TLERecord:
    """Parse a single two-line TLE into a TLERecord."""
    satrec = Satrec.twoline2rv(line1, line2, WGS72)
    if satrec.error != 0:
        raise ValueError(
            f"sgp4 failed to parse TLE (error code {satrec.error}): "
            f"line1={line1!r} line2={line2!r}"
        )

    satellite_id = str(satrec.satnum)
    return TLERecord(
        satellite_id=satellite_id,
        name=name or f"UNNAMED-{satellite_id}",
        line1=line1,
        line2=line2,
        satrec=satrec,
    )


def parse_tle_file(path: str) -> list[TLERecord]:
    """Parse a file containing one or more TLEs (2-line or 3-line format)."""
    with open(path) as f:
        raw_lines = [line.rstrip("\n") for line in f if line.strip()]

    records: list[TLERecord] = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]

        if line.startswith("1 "):
            if i + 1 >= len(raw_lines) or not raw_lines[i + 1].startswith("2 "):
                raise ValueError(
                    f"Malformed TLE at line {i + 1}: '1 ' line not "
                    f"followed by a '2 ' line"
                )
            records.append(parse_tle(line, raw_lines[i + 1]))
            i += 2
        else:
            name = line.strip()
            if (
                i + 2 >= len(raw_lines)
                or not raw_lines[i + 1].startswith("1 ")
                or not raw_lines[i + 2].startswith("2 ")
            ):
                raise ValueError(
                    f"Malformed TLE near line {i + 1}: expected a name "
                    f"line followed by '1 '/'2 ' lines"
                )
            records.append(parse_tle(raw_lines[i + 1], raw_lines[i + 2], name=name))
            i += 3

    return records
