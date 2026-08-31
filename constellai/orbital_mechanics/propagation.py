"""SGP4 propagation wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from sgp4.api import SGP4_ERRORS, jday

from .tle import TLERecord


@dataclass(frozen=True)
class StateVector:
    """A propagated state in the TEME frame."""

    satellite_id: str
    epoch: datetime
    position_km: np.ndarray
    velocity_km_s: np.ndarray


class PropagationError(RuntimeError):
    """Raised when SGP4 reports an error for the requested epoch."""


def propagate(record: TLERecord, when: datetime) -> StateVector:
    """Propagate a TLE record to a UTC datetime."""
    if when.tzinfo is not None:
        when = when.astimezone(timezone.utc).replace(tzinfo=None)

    jd, fr = jday(
        when.year,
        when.month,
        when.day,
        when.hour,
        when.minute,
        when.second + when.microsecond / 1e6,
    )
    error_code, position, velocity = record.satrec.sgp4(jd, fr)
    if error_code != 0:
        raise PropagationError(
            f"SGP4 error {error_code} for satellite {record.satellite_id} "
            f"at {when.isoformat()}: {SGP4_ERRORS[error_code]}"
        )

    return StateVector(
        satellite_id=record.satellite_id,
        epoch=when,
        position_km=np.array(position, dtype=np.float64),
        velocity_km_s=np.array(velocity, dtype=np.float64),
    )


def propagate_series(
    record: TLERecord,
    start: datetime,
    end: datetime,
    step: timedelta,
) -> list[StateVector]:
    """Propagate a satellite at fixed intervals in an exclusive-end window."""
    states = []
    current = start
    while current < end:
        states.append(propagate(record, current))
        current += step
    return states