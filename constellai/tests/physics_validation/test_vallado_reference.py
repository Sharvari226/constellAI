"""Physics validation: pin our propagation wrapper to Vallado's published
reference state vectors.

This is the M1 hard gate referenced throughout the project docs. It does
not merely check that the underlying python-sgp4 library is internally
consistent (that's already covered by `python -m sgp4.tests`, which
runs the full official 45-case verification suite bundled with the
library). Instead, this test checks that *our* wrapper -- TLE parsing,
datetime handling, and the propagate()/propagate_series() interface the
rest of the codebase will actually call -- reproduces the same published
numbers, so a bug introduced in our thin wrapper layer can't silently
slip past "the library itself is fine."

Reference case: Vallado's classic verification satellite, catalog #00005
("SGP4-VER.TLE" first entry). Expected state vectors are copied from the
official reference C++ output distributed with python-sgp4
(sgp4/tcppver.out), not re-derived by hand.
"""

from datetime import timedelta

import numpy as np
import pytest
from sgp4.conveniences import sat_epoch_datetime

from constellai.orbital_mechanics.propagation import propagate
from constellai.orbital_mechanics.tle import parse_tle

# fmt: off
LINE1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
LINE2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"

# (minutes since epoch, expected position_km (x,y,z), expected velocity_km_s (x,y,z))
# Copied verbatim from python-sgp4's bundled tcppver.out for satellite 00005.
REFERENCE_VECTORS = [
    (0.0,
     (7022.46529266, -1400.08296755, 0.03995155),
     (1.893841015, 6.405893759, 4.534807250)),
    (360.0,
     (-7154.03120202, -3783.17682504, -3536.19412294),
     (4.741887409, -4.151817765, -2.093935425)),
    (720.0,
     (-7134.59340119, 6531.68641334, 3260.27186483),
     (-4.113793027, -2.911922039, -2.557327851)),
    (1440.0,
     (-938.55923943, -6268.18748831, -4294.02924751),
     (7.536105209, -0.427127707, 0.989878080)),
]
# fmt: on

# Vallado's reference output is given to ~9 significant figures; we allow
# a small absolute tolerance to account for floating-point accumulation
# differences between the reference C++ build and this environment's
# double-precision arithmetic, not to mask a real discrepancy.
POSITION_TOLERANCE_KM = 1e-5
VELOCITY_TOLERANCE_KM_S = 1e-6


@pytest.fixture
def record():
    return parse_tle(LINE1, LINE2, name="Vallado Test Satellite 00005")


@pytest.mark.parametrize("minutes_since_epoch,expected_pos,expected_vel", REFERENCE_VECTORS)
def test_propagate_matches_vallado_reference(record, minutes_since_epoch, expected_pos, expected_vel):
    epoch = sat_epoch_datetime(record.satrec)
    when = epoch + timedelta(minutes=minutes_since_epoch)

    state = propagate(record, when)

    np.testing.assert_allclose(
        state.position_km, np.array(expected_pos), atol=POSITION_TOLERANCE_KM,
        err_msg=f"Position mismatch at t={minutes_since_epoch} min since epoch",
    )
    np.testing.assert_allclose(
        state.velocity_km_s, np.array(expected_vel), atol=VELOCITY_TOLERANCE_KM_S,
        err_msg=f"Velocity mismatch at t={minutes_since_epoch} min since epoch",
    )


def test_propagate_series_matches_individual_calls(record):
    """propagate_series() must not silently diverge from repeated propagate()
    calls -- e.g. through incorrect step accumulation or timezone handling."""
    epoch = sat_epoch_datetime(record.satrec)
    start = epoch
    end = epoch + timedelta(minutes=30)
    step = timedelta(minutes=10)

    from constellai.orbital_mechanics.propagation import propagate_series
    series = propagate_series(record, start, end, step)

    assert len(series) == 3
    for state in series:
        direct = propagate(record, state.epoch)
        np.testing.assert_allclose(state.position_km, direct.position_km, atol=1e-9)
        np.testing.assert_allclose(state.velocity_km_s, direct.velocity_km_s, atol=1e-9)