"""Unit tests for HCW relative motion.

The key physics validation here mirrors the spirit of the M1 Vallado
gate: unforced HCW motion is analytically periodic with period
T = 2*pi/n. Propagating a known initial condition forward one full
period, with zero control input, should return (approximately) to the
starting state -- this catches integration bugs the same way the M1
gate catches propagation bugs, using a known closed-form property
rather than an arbitrary tolerance.
"""

import math

import numpy as np
import pytest

from constellai.orbital_mechanics.relative_motion import (
    RelativeState,
    propagate_hcw_step,
)


def test_zero_state_zero_control_stays_at_zero():
    state = RelativeState(position=np.zeros(3), velocity=np.zeros(3))
    result = propagate_hcw_step(state, control_accel=np.zeros(3), mean_motion=0.001, dt=1.0)
    np.testing.assert_allclose(result.as_vector(), np.zeros(6), atol=1e-12)


def test_unforced_motion_is_periodic():
    """A satellite offset in the along-track direction with matching
    velocity should return to (approximately) its initial state after
    one full orbital period -- this is a known analytic property of
    unforced HCW motion, not an arbitrary check."""
    n = 0.0011  # rad/s, roughly a 95-minute LEO orbit
    period = 2 * math.pi / n

    initial = RelativeState(
        position=np.array([100.0, 0.0, 0.0]),  # 100m radial offset
        velocity=np.array([0.0, -2 * n * 100.0, 0.0]),  # matching along-track drift rate for a bounded orbit
    )

    state = initial
    n_steps = 2000
    dt = period / n_steps
    for _ in range(n_steps):
        state = propagate_hcw_step(state, control_accel=np.zeros(3), mean_motion=n, dt=dt)

    np.testing.assert_allclose(state.position, initial.position, atol=1.0)  # within 1m after a full period
    np.testing.assert_allclose(state.velocity, initial.velocity, atol=1e-3)


def test_control_acceleration_changes_trajectory():
    """Sanity check: applying thrust must actually change the resulting
    state relative to no thrust -- catches a control_accel plumbing bug
    (e.g. accidentally ignored in hcw_derivative)."""
    state = RelativeState(position=np.array([50.0, 0.0, 0.0]), velocity=np.zeros(3))
    n = 0.0011

    no_thrust = propagate_hcw_step(state, control_accel=np.zeros(3), mean_motion=n, dt=10.0)
    with_thrust = propagate_hcw_step(state, control_accel=np.array([0.01, 0.0, 0.0]), mean_motion=n, dt=10.0)

    assert not np.allclose(no_thrust.as_vector(), with_thrust.as_vector())