"""Two-line element parsing."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike

from sgp4.api import Satrec, WGS72


@dataclass(frozen=True)
class TLERecord:
    """A parsed TLE and its NORAD catalog identifier."""

    satellite_id: str
    line1: str
    line2: str
    satrec: Satrec
    name: str


def parse_tle(line1: str, line2: str, *, name: str | None = None) -> TLERecord:
    """Parse two TLE lines into the record used by propagation APIs."""
    if not line1.startswith("1 ") or not line2.startswith("2 "):
        raise ValueError("TLE must contain a line 1 followed by a line 2")

    satrec = Satrec.twoline2rv(line1, line2, WGS72)
    if satrec.error != 0:
        raise ValueError(f"sgp4 failed to parse TLE (error code {satrec.error})")

    satellite_id = str(satrec.satnum)
    return TLERecord(
        satellite_id=satellite_id,
        line1=line1,
        line2=line2,
        satrec=satrec,
        name=name or f"UNNAMED-{satellite_id}",
    )


def parse_tle_file(path: str | PathLike[str]) -> list[TLERecord]:
    """Parse a file containing one or more two-line or three-line TLEs."""
    with open(path, encoding="utf-8") as tle_file:
        lines = [line.strip() for line in tle_file if line.strip()]

    records: list[TLERecord] = []
    index = 0
    while index < len(lines):
        if lines[index].startswith("1 "):
            if index + 1 >= len(lines) or not lines[index + 1].startswith("2 "):
                raise ValueError("Malformed TLE: line 1 must be followed by line 2")
            records.append(parse_tle(lines[index], lines[index + 1]))
            index += 2
            continue

        if (
            index + 2 >= len(lines)
            or not lines[index + 1].startswith("1 ")
            or not lines[index + 2].startswith("2 ")
        ):
            raise ValueError("Malformed TLE: expected name, line 1, and line 2")
        records.append(parse_tle(lines[index + 1], lines[index + 2], name=lines[index]))
        index += 3

    return records