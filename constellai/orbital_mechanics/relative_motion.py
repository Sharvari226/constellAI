"""Hill-Clohessy-Wiltshire (HCW) equations: linearized relative motion
between a chaser satellite and a circular reference orbit, in the LVLH
frame (x=radial, y=along-track, z=cross-track).

This is a deliberate, named approximation, not a silent simplification:
HCW assumes (1) the reference orbit is circular, and (2) separation is
small relative to orbital radius. Both hold for the close-proximity,
short-horizon regime a conjunction-avoidance maneuver actually operates
in -- which is exactly why this is the standard simplification used for
this problem in the aerospace literature (see Vallado, "Fundamentals of
Astrodynamics and Applications" -- the same reference this project's M1
propagator validation gate already relies on). It would NOT be valid
for long-horizon, large-separation, or highly eccentric-reference
scenarios; M4's environment must never be used outside this regime
without that caveat being re-examined.

Used by M4 to propagate relative state cheaply at every RL timestep --
full SGP4 propagation for both satellites at every step of every
episode of every training run would make RL training intractable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RelativeState:
    """Relative position/velocity in the LVLH frame, meters and m/s.

    position : np.ndarray, shape (3,) -- [radial, along-track, cross-track]
    velocity : np.ndarray, shape (3,)
    """

    position: np.ndarray
    velocity: np.ndarray

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.position, self.velocity])

    @staticmethod
    def from_vector(v: np.ndarray) -> "RelativeState":
        return RelativeState(position=v[:3], velocity=v[3:])


def hcw_derivative(state_vec: np.ndarray, control_accel: np.ndarray, mean_motion: float) -> np.ndarray:
    """Right-hand side of the HCW ODE: d(state)/dt.

    Parameters
    ----------
    state_vec : np.ndarray, shape (6,) -- [x, y, z, vx, vy, vz]
    control_accel : np.ndarray, shape (3,) -- applied thrust acceleration, m/s^2
    mean_motion : float -- reference orbit's mean motion n, rad/s

    Returns
    -------
    np.ndarray, shape (6,)

    Notes
    -----
    The equations (Vallado, standard form):
        x'' = 3 n^2 x + 2 n y' + ax
        y'' = -2 n x' + ay
        z'' = -n^2 z + az
    """
    x, y, z, vx, vy, vz = state_vec
    ax, ay, az = control_accel
    n = mean_motion

    return np.array([
        vx, vy, vz,
        3 * n**2 * x + 2 * n * vy + ax,
        -2 * n * vx + ay,
        -n**2 * z + az,
    ])


def propagate_hcw_step(
    state: RelativeState,
    control_accel: np.ndarray,
    mean_motion: float,
    dt: float,
) -> RelativeState:
    """Advance relative state by one timestep using classical RK4.

    Parameters
    ----------
    state : RelativeState
    control_accel : np.ndarray, shape (3,) -- held constant over this step
        (zero-order hold), m/s^2
    mean_motion : float -- rad/s
    dt : float -- seconds

    Returns
    -------
    RelativeState
    """
    y0 = state.as_vector()

    k1 = hcw_derivative(y0, control_accel, mean_motion)
    k2 = hcw_derivative(y0 + dt / 2 * k1, control_accel, mean_motion)
    k3 = hcw_derivative(y0 + dt / 2 * k2, control_accel, mean_motion)
    k4 = hcw_derivative(y0 + dt * k3, control_accel, mean_motion)

    y_next = y0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return RelativeState.from_vector(y_next)