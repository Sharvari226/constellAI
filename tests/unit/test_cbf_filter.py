"""Unit tests for the HOCBF safety filter.

test_correction_actually_satisfies_the_constraint is the one that
matters most -- it verifies the corrected action isn't just "different
from proposed", it genuinely satisfies the HOCBF safety condition,
which is the entire point of this module existing.
"""

import numpy as np
import pytest

from constellai.safety.cbf_filter import hocbf_safe_action, natural_acceleration


def test_natural_acceleration_matches_hcw_derivative_at_zero_control():
    from constellai.orbital_mechanics.relative_motion import hcw_derivative
    state = np.array([100.0, 0.0, 0.0, 0.0, -0.5, 0.0])
    n = 0.0011
    expected = hcw_derivative(state, np.zeros(3), n)[3:]
    actual = natural_acceleration(state, n)
    np.testing.assert_allclose(actual, expected)


def test_safe_action_far_from_threat_passes_through_unmodified():
    """Far away, moving away -- the proposed action should never need
    correction; this is the common, uneventful case."""
    result = hocbf_safe_action(
        relative_position_m=np.array([10000.0, 0.0, 0.0]),
        relative_velocity_mps=np.array([10.0, 0.0, 0.0]),  # moving further away
        proposed_action_mps2=np.zeros(3),
        mean_motion=0.0011,
        safety_radius_m=50.0,
    )
    assert result.was_corrected is False
    np.testing.assert_allclose(result.safe_action, np.zeros(3))


def test_dangerous_proposed_action_gets_corrected():
    """Close to the threat, closing fast, and the proposed action
    (zero thrust) does nothing to prevent it -- this MUST trigger a
    correction."""
    result = hocbf_safe_action(
        relative_position_m=np.array([60.0, 0.0, 0.0]),  # just outside the 50m safety radius
        relative_velocity_mps=np.array([-5.0, 0.0, 0.0]),  # closing fast
        proposed_action_mps2=np.zeros(3),
        mean_motion=0.0011,
        safety_radius_m=50.0,
    )
    assert result.was_corrected is True
    assert not np.allclose(result.safe_action, np.zeros(3))


def test_correction_actually_satisfies_the_constraint():
    """The core claim of this whole module: after correction, the
    HOCBF safety condition (psi1_dot + alpha1*psi1 >= 0) must actually
    hold -- not just 'the action changed', but 'the action is now
    genuinely safe by this module's own definition of safe'."""
    p = np.array([60.0, 0.0, 0.0])
    v = np.array([-5.0, 0.0, 0.0])
    n = 0.0011
    alpha0, alpha1 = 1.0, 1.0
    safety_radius = 50.0

    result = hocbf_safe_action(
        relative_position_m=p, relative_velocity_mps=v,
        proposed_action_mps2=np.zeros(3), mean_motion=n,
        safety_radius_m=safety_radius, alpha0=alpha0, alpha1=alpha1,
    )
    assert result.was_corrected is True

    a_natural = natural_acceleration(np.concatenate([p, v]), n)
    h = float(np.dot(p, p) - safety_radius**2)
    h_dot = float(2 * np.dot(p, v))
    psi1 = h_dot + alpha0 * h
    psi1_dot = 2 * np.dot(v, v) + 2 * np.dot(p, a_natural + result.safe_action) + alpha0 * h_dot

    assert psi1_dot + alpha1 * psi1 >= -1e-6  # allow tiny float tolerance


def test_correction_is_minimal_deviation_along_constraint_gradient():
    """The correction vector should be PARALLEL to the constraint
    gradient (2p) -- this is what 'minimal necessary correction' means
    mathematically, not an arbitrary safe-ish nudge in any direction."""
    p = np.array([60.0, 0.0, 0.0])
    v = np.array([-5.0, 0.0, 0.0])
    proposed = np.zeros(3)

    result = hocbf_safe_action(
        relative_position_m=p, relative_velocity_mps=v,
        proposed_action_mps2=proposed, mean_motion=0.0011, safety_radius_m=50.0,
    )
    correction = result.safe_action - proposed
    gradient = 2 * p

    # Parallel vectors: cross product is zero (in 3D). Using normalized
    # dot product close to +/-1 as the parallel check.
    cos_angle = np.dot(correction, gradient) / (np.linalg.norm(correction) * np.linalg.norm(gradient))
    assert abs(cos_angle) == pytest.approx(1.0, abs=1e-6)


def test_barrier_value_negative_when_already_inside_unsafe_sphere():
    result = hocbf_safe_action(
        relative_position_m=np.array([10.0, 0.0, 0.0]),  # already inside the 50m radius
        relative_velocity_mps=np.zeros(3),
        proposed_action_mps2=np.zeros(3),
        mean_motion=0.0011,
        safety_radius_m=50.0,
    )
    assert result.barrier_value < 0