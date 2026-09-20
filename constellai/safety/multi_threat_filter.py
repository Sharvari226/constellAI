"""M5 completion: multi-threat HOCBF safety filter.

The single-threat closed-form solution (cbf_filter.py) generalizes to
zero or one active constraint only -- each threat demands a correction
along its OWN constraint gradient direction, and with multiple
simultaneous threats these directions can conflict. There is no closed
form once more than one constraint is active; this is a genuine QP:

    minimize    ||u - u_proposed||^2
    subject to  2*p_k . u >= b_k   for every threat k
                |u_i| <= u_max     (physical thrust limit)

Solved via scipy's SLSQP -- the standard, minimal-dependency choice for
a small (few-constraint, 3-variable) QP at each control step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from constellai.orbital_mechanics.relative_motion import hcw_derivative
from constellai.safety.cbf_filter import DEFAULT_ALPHA0, DEFAULT_ALPHA1, DEFAULT_SAFETY_RADIUS_M


@dataclass(frozen=True)
class Threat:
    relative_position_m: np.ndarray
    relative_velocity_mps: np.ndarray
    safety_radius_m: float = DEFAULT_SAFETY_RADIUS_M


@dataclass(frozen=True)
class MultiThreatFilterResult:
    safe_action: np.ndarray
    was_corrected: bool
    barrier_values: list[float]
    solver_success: bool


def _hocbf_constraint_terms(
    p: np.ndarray, v: np.ndarray, safety_radius_m: float, mean_motion: float,
    alpha0: float, alpha1: float,
) -> tuple[np.ndarray, float, float]:
    """Returns (gradient, b, h) for one threat's linear-in-u HOCBF
    constraint 2*p.u >= b. h computed here, ONCE, with the actual
    radius already included -- no deferred/patched-later radius term,
    which is exactly what produced a real bug in an earlier version of
    this function (an incorrect after-the-fact correction term)."""
    a_natural = hcw_derivative(np.concatenate([p, v]), np.zeros(3), mean_motion)[3:]
    h = float(np.dot(p, p) - safety_radius_m**2)
    h_dot = float(2 * np.dot(p, v))
    psi1 = h_dot + alpha0 * h
    b = -(2 * np.dot(v, v) + 2 * np.dot(p, a_natural) + alpha0 * h_dot) - alpha1 * psi1
    return 2 * p, b, h


def multi_threat_safe_action(
    threats: list[Threat],
    proposed_action_mps2: np.ndarray,
    mean_motion: float,
    max_thrust_mps2: float,
    alpha0: float = DEFAULT_ALPHA0,
    alpha1: float = DEFAULT_ALPHA1,
) -> MultiThreatFilterResult:
    """Filter one proposed action against ALL simultaneous threats."""
    if not threats:
        return MultiThreatFilterResult(
            safe_action=proposed_action_mps2, was_corrected=False,
            barrier_values=[], solver_success=True,
        )

    constraints = []
    barrier_values = []
    for threat in threats:
        p, v, r = threat.relative_position_m, threat.relative_velocity_mps, threat.safety_radius_m
        grad, b, h = _hocbf_constraint_terms(p, v, r, mean_motion, alpha0, alpha1)
        barrier_values.append(h)
        constraints.append({"type": "ineq", "fun": (lambda u, g=grad, bb=b: np.dot(g, u) - bb)})

    all_satisfied = all(c["fun"](proposed_action_mps2) >= -1e-9 for c in constraints)
    if all_satisfied:
        return MultiThreatFilterResult(
            safe_action=proposed_action_mps2, was_corrected=False,
            barrier_values=barrier_values, solver_success=True,
        )

    bounds = [(-max_thrust_mps2, max_thrust_mps2)] * 3

    def objective(u):
        diff = u - proposed_action_mps2
        return float(np.dot(diff, diff))

    result = minimize(
        objective, x0=proposed_action_mps2, method="SLSQP",
        bounds=bounds, constraints=constraints,
    )

    return MultiThreatFilterResult(
        safe_action=result.x if result.success else proposed_action_mps2,
        was_corrected=result.success,
        barrier_values=barrier_values,
        solver_success=result.success,
    )