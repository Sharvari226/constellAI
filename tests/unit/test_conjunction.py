"""Unit tests for conjunction geometry (TCA + miss distance)."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from constellai.orbital_mechanics.conjunction import screen_pair
from constellai.orbital_mechanics.propagation import StateVector


def _state(sat_id, t, pos, vel=(0.0, 0.0, 0.0)):
    return StateVector(
        satellite_id=sat_id,
        epoch=t,
        position_km=np.array(pos, dtype=np.float64),
        velocity_km_s=np.array(vel, dtype=np.float64),
    )


def test_screen_pair_finds_true_minimum():
    """Two satellites closing then separating -- TCA must land on the
    actual minimum-separation sample, not the first or last."""
    t0 = datetime(2026, 1, 1)
    times = [t0 + timedelta(minutes=i) for i in range(5)]

    states_a = [_state("A", t, (0, 0, 0)) for t in times]
    states_b = [
        _state("B", times[0], (5, 0, 0)),
        _state("B", times[1], (3, 0, 0)),
        _state("B", times[2], (1, 0, 0)),
        _state("B", times[3], (3, 0, 0)),
        _state("B", times[4], (5, 0, 0)),
    ]

    event = screen_pair(states_a, states_b)

    assert event.tca == times[2]
    assert event.miss_distance_km == pytest.approx(1.0)
    assert event.satellite_id_a == "A"
    assert event.satellite_id_b == "B"


def test_screen_pair_computes_relative_speed():
    t0 = datetime(2026, 1, 1)
    states_a = [_state("A", t0, (0, 0, 0), vel=(1, 0, 0))]
    states_b = [_state("B", t0, (10, 0, 0), vel=(-1, 0, 0))]

    event = screen_pair(states_a, states_b)

    assert event.relative_speed_km_s == pytest.approx(2.0)


def test_screen_pair_rejects_mismatched_lengths():
    t0 = datetime(2026, 1, 1)
    states_a = [_state("A", t0, (0, 0, 0))]
    states_b = [_state("B", t0, (1, 0, 0)), _state("B", t0, (2, 0, 0))]

    with pytest.raises(ValueError):
        screen_pair(states_a, states_b)


def test_screen_pair_rejects_empty_input():
    with pytest.raises(ValueError):
        screen_pair([], [])


def test_screen_pair_rejects_misaligned_timestamps():
    t0 = datetime(2026, 1, 1)
    t1 = datetime(2026, 1, 2)
    states_a = [_state("A", t0, (0, 0, 0))]
    states_b = [_state("B", t1, (1, 0, 0))]

    with pytest.raises(ValueError):
        screen_pair(states_a, states_b)