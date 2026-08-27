"""SGP4 propagation wrapper.

This module is intentionally the only place in the codebase that calls
into python-sgp4 to advance a satellite's state forward in time. It
returns state vectors in the TEME (True Equator, Mean Equinox) frame,
which is SGP4's native output frame -- callers needing other frames
(e.g. ECEF) must go through an explicit transform (see frames.py, not
yet implemented), never assume TEME == any other frame.

Units: position in km, velocity in km/s -- python-sgp4's native units,
kept as-is here rather than silently converted, so unit bugs can't hide
at this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from sgp4.api import Satrec, SGP4_ERRORS, jday

from constellai.orbital_mechanics.tle import TLERecord


@dataclass(frozen=True)
class StateVector:
    """A single propagated satellite state at one instant.

    Attributes
    ----------
    satellite_id : str
        NORAD catalog number, matching the originating TLERecord.
    epoch : datetime
        UTC timestamp this state corresponds to.
    position_km : np.ndarray, shape (3,)
        Position in the TEME frame, kilometers.
    velocity_km_s : np.ndarray, shape (3,)
        Velocity in the TEME frame, kilometers/second.
    """

    satellite_id: str
    epoch: datetime
    position_km: np.ndarray
    velocity_km_s: np.ndarray


class PropagationError(RuntimeError):
    """Raised when SGP4 reports an internal propagation error.

    SGP4 error codes (1-6) indicate physically meaningful failure modes
    (e.g. decayed orbit, eccentricity out of range) -- these are
    re-raised with the human-readable message rather than silently
    returning NaNs, so a bad propagation can never masquerade as a
    valid state downstream.
    """


def propagate(record: TLERecord, when: datetime) -> StateVector:
    """Propagate a single satellite to a specific UTC datetime.

    Parameters
    ----------
    record : TLERecord
        The parsed TLE to propagate.
    when : datetime
        Target time. Must be timezone-aware UTC (or naive, assumed UTC).

    Returns
    -------
    StateVector

    Raises
    ------
    PropagationError
        If SGP4 reports an error code for this epoch (e.g. the requested
        time is far enough from TLE epoch that the analytic model is
        known to break down).
    """
    if when.tzinfo is not None:
        when = when.astimezone(timezone.utc).replace(tzinfo=None)

    jd, fr = jday(when.year, when.month, when.day, when.hour, when.minute,
                  when.second + when.microsecond / 1e6)

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
    """Propagate a satellite across a time window at fixed intervals.

    Parameters
    ----------
    record : TLERecord
        The parsed TLE to propagate.
    start, end : datetime
        Inclusive start, exclusive end of the propagation window (UTC).
    step : timedelta
        Interval between samples.

    Returns
    -------
    list[StateVector]
        One state per sample point, in chronological order.

    Notes
    -----
    This fixed-interval sampling is intended for M1/M6 validation and for
    the non-ML baseline (Step 3 of the roadmap). M2's graph construction
    and M3's continuous-time forecasting operate on event-driven sampling
    instead, not this function -- do not reuse this for those modules
    without an explicit design discussion, since fixed-interval sampling
    is exactly the snapshot-based limitation the project deliberately
    avoids for the forecasting stage.
    """
    states = []
    t = start
    while t < end:
        states.append(propagate(record, t))
        t += step
    return states