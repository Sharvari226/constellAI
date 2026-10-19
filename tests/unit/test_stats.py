# Tests for the Wilson interval used in all M6 safety-rate reporting.

import pytest

from constellai.evaluation.stats import wilson_interval


def test_zero_successes_matches_reference_values():
    # standard reference: 0/10 at 95% -> upper bound about 0.278
    low, high = wilson_interval(0, 10)
    assert low == 0.0
    assert high == pytest.approx(0.2775, abs=1e-3)


def test_all_successes_matches_reference_values():
    low, high = wilson_interval(10, 10)
    assert low == pytest.approx(0.7225, abs=1e-3)
    assert high == 1.0


def test_interval_contains_the_estimate_and_stays_in_bounds():
    for successes in (0, 1, 250, 487, 500):
        low, high = wilson_interval(successes, 500)
        assert 0.0 <= low <= successes / 500 <= high <= 1.0


def test_more_trials_give_a_tighter_interval():
    low_small, high_small = wilson_interval(45, 50)
    low_large, high_large = wilson_interval(450, 500)
    assert (high_large - low_large) < (high_small - low_small)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)