"""M5: physics-based safety filter, High-Order Control Barrier Function
(HOCBF) style -- a classical, closed-form CBF filter for a SINGLE
pairwise threat, not a reimplementation of GCBF+.

Honest scope statement, not a hedge: GCBF+ (Zhang et al., IEEE T-RO
2025) is a GNN-parameterized CBF with a formal safety proof, scaling to
1000+ simultaneous agents. This module is the classical, single-
constraint CBF baseline the research spec always called for comparing
AGAINST -- it is not, and does not attempt to be, a from-scratch
version of GCBF+ itself. Benchmarking this against an actual GCBF+
implementation is separate future work, not done here.

Why HOCBF, not plain CBF: the safety set is h(p) = ||p||^2 - r_safe^2
>= 0 (stay outside a collision sphere), but control (thrust) only
enters through ACCELERATION -- h depends on position, and control
doesn't appear until the second time-derivative. This relative-degree-2
structure is exactly why a plain (relative-degree-1) CBF's constraint
would be undefined here, and why the research spec specifically called
for HOCBF, following the same construction Parikh et al. use for
satellite-servicing collision avoidance.

For a SINGLE constraint, the resulting HOCBF-QP has a closed-form
solution -- no QP solver dependency needed at this scope: if the
proposed action already satisfies the constraint, it passes through
unchanged; if not, it is projected onto the constraint boundary along
the constraint's own gradient, which is precisely "the minimal
necessary correction" (Parikh et al.'s own framing, and this project's
explicit "never claim guaranteed safety, report the correction rate"
commitment). Handling MULTIPLE simultaneous constraints (this
satellite threatened by several others in the same step) needs a real
QP solver and is a disclosed, real limitation of this module as it
stands -- exactly the same category of limitation Parikh et al.
disclose for their own multi-constraint case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constellai.orbital_mechanics.relative_motion import hcw_derivative

DEFAULT_SAFETY_RADIUS_M = 50.0  # matches DEFAULT_COLLISION_RADIUS_M in environment.py
DEFAULT_ALPHA0 = 1.0  # HOCBF class-K gain for psi0 = h
DEFAULT_ALPHA1 = 1.0  # HOCBF class-K gain for psi1


@dataclass(frozen=True)
class SafetyFilterResult:
    """Outcome of filtering one proposed action through the HOCBF check.

    safe_action : np.ndarray, shape (3,)
        The action actually safe to execute -- equals proposed_action
        unchanged if it already satisfied the constraint.
    was_corrected : bool
        Whether the safety filter actually changed the proposed action.
        This is the number to report honestly (correction rate), not
        "how often did the system prevent a collision" framed as a
        guarantee -- see module docstring.
    barrier_value : float
        h(p) at the current state -- negative means ALREADY inside the
        unsafe sphere (a genuine failure state upstream; the filter can
        still act, but this is a signal worth logging distinctly from
        an ordinary near-boundary correction).
    """

    safe_action: np.ndarray
    was_corrected: bool
    barrier_value: float


def natural_acceleration(state_vec: np.ndarray, mean_motion: float) -> np.ndarray:
    """The HCW acceleration a satellite experiences with ZERO control
    input -- reuses hcw_derivative directly rather than re-deriving the
    natural/Coriolis terms separately, so this can never silently drift
    out of sync with the validated propagation code in
    relative_motion.py."""
    derivative = hcw_derivative(state_vec, np.zeros(3), mean_motion)
    return derivative[3:]


def hocbf_safe_action(
    relative_position_m: np.ndarray,
    relative_velocity_mps: np.ndarray,
    proposed_action_mps2: np.ndarray,
    mean_motion: float,
    safety_radius_m: float = DEFAULT_SAFETY_RADIUS_M,
    alpha0: float = DEFAULT_ALPHA0,
    alpha1: float = DEFAULT_ALPHA1,
) -> SafetyFilterResult:
    """Filter one proposed action for one pairwise threat via HOCBF.

    Parameters
    ----------
    relative_position_m, relative_velocity_mps : np.ndarray, shape (3,)
        This satellite's state relative to the threat.
    proposed_action_mps2 : np.ndarray, shape (3,)
        The maneuver an upstream policy (any M4 rung) wants to execute.
    mean_motion : float, rad/s -- reference orbit's mean motion, needed
        to evaluate the natural (Coriolis + centrifugal) HCW acceleration
        this satellite experiences even with zero control.
    safety_radius_m, alpha0, alpha1 : float, optional
        HOCBF parameters -- same placeholder status as every other
        threshold in this project, not yet through Step-1 spec freeze.

    Returns
    -------
    SafetyFilterResult

    Notes
    -----
    Derivation: h(p) = ||p||^2 - r_safe^2.
        psi0 = h
        psi1 = h_dot + alpha0 * psi0 = 2*p.v + alpha0*(||p||^2 - r_safe^2)
    Safety condition: psi1_dot + alpha1*psi1 >= 0, where
        psi1_dot = 2||v||^2 + 2*p.(a_natural + u) + alpha0*h_dot
    This is LINEAR in the control input u -- rearranged to
        2*p . u >= b
    for a scalar b gathering every other term, giving a single linear
    inequality constraint on u. The minimal-deviation solution to
    "minimize ||u - u_proposed||^2 subject to 2p.u >= b" has the
    standard closed form used below: project onto the constraint
    boundary along the constraint gradient (2p) only if the proposed
    action violates it.
    """
    p = relative_position_m
    v = relative_velocity_mps
    a_natural = natural_acceleration(np.concatenate([p, v]), mean_motion)

    h = float(np.dot(p, p) - safety_radius_m**2)
    h_dot = float(2 * np.dot(p, v))
    psi1 = h_dot + alpha0 * h

    constraint_gradient = 2 * p  # d(psi1_dot)/du, since psi1_dot = 2*p.u + (terms not involving u)
    b = -(2 * np.dot(v, v) + 2 * np.dot(p, a_natural) + alpha0 * h_dot) - alpha1 * psi1

    current_value = float(np.dot(constraint_gradient, proposed_action_mps2))

    if current_value >= b:
        # Proposed action already satisfies the HOCBF condition -- pass
        # through unmodified. This is the common case when there is no
        # genuine imminent threat.
        return SafetyFilterResult(safe_action=proposed_action_mps2, was_corrected=False, barrier_value=h)

    grad_norm_sq = float(np.dot(constraint_gradient, constraint_gradient))
    if grad_norm_sq < 1e-12:
        # Degenerate case: p is (numerically) zero -- constraint
        # gradient vanishes, no well-defined minimal correction exists
        # via this closed form. This is itself a meaningful failure
        # signal (already at/past the threat's exact position), not
        # something to paper over with an arbitrary fallback action.
        return SafetyFilterResult(safe_action=proposed_action_mps2, was_corrected=False, barrier_value=h)

    correction = constraint_gradient * (b - current_value) / grad_norm_sq
    safe_action = proposed_action_mps2 + correction

    return SafetyFilterResult(safe_action=safe_action, was_corrected=True, barrier_value=h)