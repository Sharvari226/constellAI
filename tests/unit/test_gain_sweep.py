# Tests for the Phase 4b gain-selection procedure.
#
# Most tests build result stores by hand so the selection rules (one-standard-error
# tie-break, do-no-harm rejection, no-gain-passes) are checked exactly, independent
# of any simulation.

import math

import pytest

from constellai.evaluation.gain_sweep import (
    STREAM_VALIDATION,
    baseline_free_rate,
    paired_effect,
    run_sweep,
    select_gain,
    select_policies,
    summarize_gain,
)
from constellai.evaluation.monte_carlo import cell_key, load_results
from constellai.evaluation.scenarios import STREAM_EVAL, STREAM_TRAIN, make_rng, sample_scenario
from constellai.evaluation.tiers import TIERS
from constellai.evaluation.trials import ConstantBurnPolicy, ZeroPolicy

N = 100


def _cell(free, bits=None, correction=0.0, fuel=0.0):
    if bits is None:
        bits = "0" * free + "1" * (N - free)
    return {
        "n_trials": N, "collision_free": free, "collision_free_rate": free / N,
        "collided_bits": bits, "filter_correction_rate": correction, "mean_fuel_used_per_agent": fuel,
    }


def _store(per_alpha, off_free, tiers=("nominal",), labels=("p",)):
    # per_alpha: {alpha: (free, correction)}
    cells = {}
    for tier in tiers:
        for label in labels:
            cells[cell_key(tier, "hcw", None, label, N, 0)] = _cell(off_free)
            for alpha, (free, correction) in per_alpha.items():
                cells[cell_key(tier, "hcw", alpha, label, N, 0)] = _cell(free, correction=correction)
    return {"meta": {}, "cells": cells}


def _rows(per_alpha, off_free, **kwargs):
    store = _store(per_alpha, off_free, **kwargs)
    tiers = kwargs.get("tiers", ("nominal",))
    labels = kwargs.get("labels", ("p",))
    return [summarize_gain(store, a, tiers, labels, "hcw", N, 0) for a in per_alpha]


def test_validation_stream_is_disjoint_from_training_and_evaluation():
    cfg = TIERS["nominal"]
    val, _ = sample_scenario(make_rng(STREAM_VALIDATION, 0, 0), cfg)
    for other in (STREAM_TRAIN, STREAM_EVAL):
        other_scenario, _ = sample_scenario(make_rng(other, 0, 0), cfg)
        assert not all((a == b).all() for a, b in zip(val, other_scenario))
    assert STREAM_VALIDATION not in (STREAM_TRAIN, STREAM_EVAL)


def test_paired_effect_counts_rescued_and_hurt_trials():
    off = {"collided_bits": "1100"}
    on = {"collided_bits": "0110"}
    # trial 0: rescued; trial 1: still collided; trial 2: hurt; trial 3: still free
    assert paired_effect(off, on) == (1, 1)


def test_summary_computes_rate_standard_error_and_fuel():
    store = _store({0.03: (80, 0.02)}, off_free=50, tiers=("nominal", "short_notice"), labels=("a", "b"))
    row = summarize_gain(store, 0.03, ("nominal", "short_notice"), ("a", "b"), "hcw", N, 0)
    assert row["mean_free_rate"] == pytest.approx(0.8)
    # four independent cells with p = 0.8, n = 100
    assert row["se"] == pytest.approx(math.sqrt(4 * 0.8 * 0.2 / N) / 4)
    assert row["mean_correction_rate"] == pytest.approx(0.02)


def test_missing_cell_is_reported():
    store = _store({0.03: (80, 0.02)}, off_free=50)
    with pytest.raises(ValueError):
        summarize_gain(store, 0.5, ("nominal",), ("p",), "hcw", N, 0)


def test_baseline_rate_comes_from_the_filter_off_cells():
    store = _store({0.03: (80, 0.02)}, off_free=50)
    assert baseline_free_rate(store, ("nominal",), ("p",), "hcw", N, 0) == pytest.approx(0.5)


def test_selection_prefers_the_least_intrusive_gain_within_one_standard_error():
    # best is 0.003 at 0.88 (SE about 0.033, threshold 0.847); 0.03 at 0.85 is within it and corrects far less
    rows = _rows({0.003: (88, 0.20), 0.03: (85, 0.02), 0.3: (55, 0.0)}, off_free=50)
    selection = select_gain(rows)
    assert selection["best_alpha"] == 0.003
    assert selection["chosen"] == 0.03
    assert 0.3 not in selection["candidates"]


def test_selection_takes_the_best_gain_when_nothing_else_is_close():
    rows = _rows({0.003: (95, 0.20), 0.03: (70, 0.02)}, off_free=50)
    assert select_gain(rows)["chosen"] == 0.003


def test_do_no_harm_rejects_a_gain_that_hurts_a_cell_and_falls_back():
    # 0.03 is within one SE of the best and the less intrusive, so it is tried first,
    # but it leaves the fragile policy worse than filter-off: it must be rejected
    cells = {
        cell_key("nominal", "hcw", None, "good", N, 0): _cell(50),
        cell_key("nominal", "hcw", None, "fragile", N, 0): _cell(90),
    }
    for alpha, good, fragile, corr in ((0.03, 99, 84, 0.02), (0.003, 95, 90, 0.20)):
        cells[cell_key("nominal", "hcw", alpha, "good", N, 0)] = _cell(good, correction=corr)
        cells[cell_key("nominal", "hcw", alpha, "fragile", N, 0)] = _cell(fragile, correction=corr)
    store = {"meta": {}, "cells": cells}
    rows = [summarize_gain(store, a, ("nominal",), ("good", "fragile"), "hcw", N, 0) for a in (0.03, 0.003)]
    selection = select_gain(rows)
    assert rows[0]["harmed_cells"] and not rows[1]["harmed_cells"]
    assert selection["best_alpha"] == 0.003
    assert selection["rejected_for_harm"] == [0.03]
    assert selection["chosen"] == 0.003


def test_no_gain_is_chosen_when_every_candidate_does_harm():
    cells = {cell_key("nominal", "hcw", None, "p", N, 0): _cell(90)}
    cells[cell_key("nominal", "hcw", 0.03, "p", N, 0)] = _cell(70, correction=0.1)
    store = {"meta": {}, "cells": cells}
    rows = [summarize_gain(store, 0.03, ("nominal",), ("p",), "hcw", N, 0)]
    selection = select_gain(rows)
    assert selection["chosen"] is None
    assert selection["rejected_for_harm"] == [0.03]


def test_harm_tolerance_allows_small_noise():
    cells = {cell_key("nominal", "hcw", None, "p", N, 0): _cell(90)}
    cells[cell_key("nominal", "hcw", 0.03, "p", N, 0)] = _cell(88, correction=0.1)  # two trials worse
    store = {"meta": {}, "cells": cells}
    assert not summarize_gain(store, 0.03, ("nominal",), ("p",), "hcw", N, 0, harm_tolerance=2)["harmed_cells"]
    assert summarize_gain(store, 0.03, ("nominal",), ("p",), "hcw", N, 0, harm_tolerance=1)["harmed_cells"]


def test_select_policies_matches_glob_patterns():
    policies = [("zero_thrust", 1), ("constant_burn", 2), ("mappo_s0_trained", 3), ("ippo_s0_untrained", 4),
                ("mappo_s0_untrained", 5)]
    labels = [l for l, _ in select_policies(policies, ["zero_thrust", "mappo_*_untrained", "*_trained"])]
    assert labels == ["zero_thrust", "mappo_s0_trained", "mappo_s0_untrained"]


def test_sweep_runs_on_the_validation_stream_and_resumes(tmp_path):
    store = tmp_path / "sweep.json"
    policies = [("zero_thrust", ZeroPolicy()), ("constant_burn", ConstantBurnPolicy())]
    seen = []
    results = run_sweep(policies, ["nominal"], "hcw", [0.03, 1.0], 10, 0, store, progress=lambda k, c: seen.append(k))
    assert len(seen) == 2 * 3  # per policy: filter off + two gains
    assert all(cell["stream"] == STREAM_VALIDATION for cell in results["cells"].values())

    seen.clear()
    run_sweep(policies, ["nominal"], "hcw", [0.03, 1.0], 10, 0, store, progress=lambda k, c: seen.append(k))
    assert seen == []  # everything already stored
    assert len(load_results(store)["cells"]) == 6


def test_a_calibrated_gain_beats_the_frozen_one_on_a_do_nothing_policy(tmp_path):
    store = tmp_path / "sweep.json"
    results = run_sweep([("zero_thrust", ZeroPolicy())], ["nominal"], "hcw", [0.03, 1.0], 40, 0, store)
    rows = [summarize_gain(results, a, ("nominal",), ("zero_thrust",), "hcw", 40, 0) for a in (0.03, 1.0)]
    assert rows[0]["mean_free_rate"] > rows[1]["mean_free_rate"] + 0.1
    assert select_gain(rows)["chosen"] == 0.03