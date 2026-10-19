# Tests for the Monte Carlo driver: cells, resume storage and the report.

import pytest

from constellai.evaluation.envs import make_env
from constellai.evaluation.monte_carlo import (
    cell_key,
    filter_token,
    format_report,
    load_results,
    parse_filter_token,
    run_cell,
    save_results,
)
from constellai.evaluation.tiers import TIERS
from constellai.evaluation.trials import ConstantBurnPolicy, ZeroPolicy, run_trials, summarize_trials


def test_filter_tokens_roundtrip():
    assert filter_token(None) == "off"
    assert filter_token(0.03) == "a0.03"
    assert filter_token(1.0) == "a1"
    assert parse_filter_token("off") is None
    assert parse_filter_token("0.03") == 0.03


def test_cell_keys_are_unique_per_setting():
    base = ("nominal", "hcw", None, "zero_thrust", 50, 0)
    keys = {
        cell_key(*base),
        cell_key("short_notice", "hcw", None, "zero_thrust", 50, 0),
        cell_key("nominal", "nonlinear", None, "zero_thrust", 50, 0),
        cell_key("nominal", "hcw", 1.0, "zero_thrust", 50, 0),
        cell_key("nominal", "hcw", 0.03, "zero_thrust", 50, 0),
        cell_key("nominal", "hcw", None, "constant_burn", 50, 0),
        cell_key("nominal", "hcw", None, "zero_thrust", 500, 0),
        cell_key("nominal", "hcw", None, "zero_thrust", 50, 1),
    }
    assert len(keys) == 8


def test_cell_matches_a_direct_trial_run():
    cell = run_cell(ZeroPolicy(), "zero_thrust", "nominal", "hcw", None, n_trials=30, eval_seed=0)
    cfg = TIERS["nominal"]
    direct = summarize_trials(run_trials(make_env("hcw", cfg, False).env, ZeroPolicy(), cfg, 30, 0))
    assert cell["collision_free"] == direct["collision_free"]
    assert len(cell["collided_bits"]) == 30
    assert cell["collided_bits"].count("0") == cell["collision_free"]
    assert "filter_correction_rate" not in cell


def test_filter_cell_reports_filter_statistics():
    cell = run_cell(ConstantBurnPolicy(), "constant_burn", "short_notice", "nonlinear", 0.03, n_trials=10)
    assert cell["filter"] == "a0.03"
    assert cell["filter_live_actions"] > 0
    assert 0.0 <= cell["filter_correction_rate"] <= 1.0
    assert cell["filter_mean_solve_ms"] >= 0.0


def test_results_roundtrip_and_resume(tmp_path):
    path = tmp_path / "results.json"
    assert load_results(path) == {"meta": {}, "cells": {}}
    cell = run_cell(ZeroPolicy(), "zero_thrust", "nominal", "hcw", None, n_trials=10)
    results = {"meta": {"note": "x"}, "cells": {cell_key("nominal", "hcw", None, "zero_thrust", 10, 0): cell}}
    save_results(path, results)
    assert load_results(path) == results
    assert not path.with_suffix(".tmp").exists()


def test_report_lists_every_policy_and_filter_column():
    cells = {}
    for alpha in (None, 0.03):
        cell = run_cell(ZeroPolicy(), "zero_thrust", "nominal", "hcw", alpha, n_trials=10)
        cells[cell_key("nominal", "hcw", alpha, "zero_thrust", 10, 0)] = cell
    report = format_report(
        {"cells": cells}, 10, 0, ["nominal"], ["hcw"], ["zero_thrust", "constant_burn"], [None, 0.03],
    )
    assert "tier: nominal" in report
    assert "filter off" in report and "filter a0.03" in report
    assert "zero_thrust" in report and "constant_burn" in report


def test_build_policies_pairs_each_checkpoint_with_its_untrained_init(tmp_path):
    pytest.importorskip("torch")
    from constellai.evaluation.monte_carlo import build_policies
    from constellai.models.marl.checkpoint import build_networks, save_checkpoint

    networks = build_networks("mappo", 3, 1.0)
    save_checkpoint(tmp_path / "mappo_seed0.pt", "mappo", 3, 1.0, networks, meta={"seed": 0, "scaled_env": True})
    save_checkpoint(tmp_path / "old_seed0.pt", "mappo", 3, 0.01, networks, meta={"seed": 0})  # unscaled: ignored
    labels = [label for label, _ in build_policies(tmp_path)]
    assert labels == ["zero_thrust", "constant_burn", "mappo_s0_untrained", "mappo_s0_trained"]