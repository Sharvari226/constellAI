"""Linear Kalman filter for one ego-threat pair, in the LVLH frame.

State x = [position(3), velocity(3)] of the THREAT relative to the EGO
(threat minus ego), meters and m/s. Both bodies obey HCW with their own
control, so the relative state obeys the same HCW with control
(u_threat - u_ego). The ego knows its own applied thrust; the threat's thrust
is unknown and enters as process noise.

Phi and B are obtained by probing propagate_hcw_step, so they match the
environment's integrator exactly (valid because one step is affine in
(state, control) for fixed dt).
"""

from __future__ import annotations

import numpy as np

from constellai.orbital_mechanics.relative_motion import RelativeState, propagate_hcw_step

_H = np.hstack([np.eye(3), np.zeros((3, 3))])  # position-only measurement


def hcw_transition(mean_motion: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (Phi, B) with x_next = Phi @ x + B @ u for one propagate_hcw_step."""
    zero3 = np.zeros(3)
    phi = np.zeros((6, 6))
    for j in range(6):
        e = np.zeros(6)
        e[j] = 1.0
        phi[:, j] = propagate_hcw_step(RelativeState.from_vector(e), zero3, mean_motion, dt).as_vector()
    b = np.zeros((6, 3))
    zero_state = RelativeState(position=np.zeros(3), velocity=np.zeros(3))
    for j in range(3):
        u = np.zeros(3)
        u[j] = 1.0
        b[:, j] = propagate_hcw_step(zero_state, u, mean_motion, dt).as_vector()
    return phi, b


class PairFilter:
    def __init__(self, mean_motion: float, dt: float, sigma_p: float,
                 process_accel_std: float, sigma_v0: float = 1.0):
        if sigma_p <= 0.0:
            raise ValueError("sigma_p must be > 0")
        self.phi, self.b = hcw_transition(mean_motion, dt)
        self.r = (sigma_p ** 2) * np.eye(3)
        self.q = (process_accel_std ** 2) * (self.b @ self.b.T)  # 0 for a known (primary) threat
        self.p0 = np.diag([sigma_p ** 2] * 3 + [sigma_v0 ** 2] * 3)
        self.mean = np.zeros(6)
        self.cov = self.p0.copy()

    def init(self, true_state: np.ndarray, rng: np.random.Generator) -> None:
        """mean ~ N(true_state, P0), cov = P0, so the filter is consistent at t=0."""
        chol = np.linalg.cholesky(self.p0)
        self.mean = np.asarray(true_state, dtype=float) + chol @ rng.standard_normal(6)
        self.cov = self.p0.copy()

    def predict(self, known_rel_accel: np.ndarray) -> None:
        """known_rel_accel = (known threat accel) - (ego applied accel), m/s^2.
        For a neighbour this is -u_ego; for the primary it is also -u_ego."""
        self.mean = self.phi @ self.mean + self.b @ np.asarray(known_rel_accel, dtype=float)
        self.cov = self.phi @ self.cov @ self.phi.T + self.q

    def update(self, z_pos: np.ndarray) -> None:
        s = _H @ self.cov @ _H.T + self.r
        k = np.linalg.solve(s, _H @ self.cov).T  # P H^T S^-1
        self.mean = self.mean + k @ (np.asarray(z_pos, dtype=float) - _H @ self.mean)
        ikh = np.eye(6) - k @ _H
        cov = ikh @ self.cov @ ikh.T + k @ self.r @ k.T  # Joseph form
        self.cov = 0.5 * (cov + cov.T)