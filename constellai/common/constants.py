"""Centralized physical constants for ConstellAI.

All orbital-mechanics constants live here, once, with their source
cited. Every other module imports from this file rather than typing
a number inline -- physics code with untracked magic numbers is a
silent bug factory (mixing WGS-72 and WGS-84 Earth radii, for
example, differ by only ~1 km but compound into meaningfully wrong
propagation over multi-day forecasts).
"""

# WGS-72 Earth gravitational parameter, km^3/s^2 -- the reference model
# SGP4 was originally fit to. Do not substitute a WGS-84 value here;
# that mismatch is a known, subtle source of propagation error.
GM_EARTH_WGS72_KM3_S2 = 398600.8

# Earth's equatorial radius, WGS-72, km.
EARTH_RADIUS_WGS72_KM = 6378.135

# J2 zonal harmonic coefficient (Earth's oblateness), WGS-72, dimensionless.
# Referenced here for documentation/future use; SGP4 itself applies this
# internally during propagation -- this project does not reimplement it.
J2_WGS72 = 1.082616e-3

SECONDS_PER_MINUTE = 60.0

# --- Placeholders, not yet scientifically finalized ---
# Both of the following are used by later modules (conjunction screening,
# graph edge inclusion) but have NOT been through the Step-1 research-spec
# freeze or any validation against real data yet. Treat any result that
# depends on these as provisional until they're deliberately set.

# Miss-distance threshold, km, below which a pair is flagged as a
# conjunction risk.
DEFAULT_SCREENING_DISTANCE_KM = 5.0

# Minimum relative closing speed, km/s, below which a pair is NOT treated
# as a genuine risk even if miss distance is small (they're close but not
# approaching each other meaningfully).
DEFAULT_MIN_CLOSING_RATE_KM_S = 0.001