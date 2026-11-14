# ConstellAI

**Forecast-conditioned, fuel-aware maneuver triggers for satellite constellations under uncertain tracks.**

Research code and experimental record for a B.Tech capstone. Everything below was read from the source, tests and result files in this repository; items that are not verified are marked **[unverified]**, and anything not done is marked **pending**.

Status tags: **done** = implemented and exercised by tests or recorded runs; **partial** = implemented, not fully evaluated; **pending** = not done.

---

## 1. Overview

**Problem.** Close approaches between satellites are growing faster than operators can triage them. Given noisy tracks of a threatening object, *when* should a satellite maneuver, and does a better risk signal reduce the fuel (Δv) needed to stay safe?

**Research question.** When does forecast quality (an uncertainty-aware, context-calibrated risk probability) reduce the Δv needed for risk-triggered maneuvers, and which calibration measure predicts it?

**Approach.** A controlled simulation: Hill–Clohessy–Wiltshire (HCW) relative motion, Kalman-filtered tracks, an analytic Monte Carlo forecaster, and a **trigger ladder**: policies that share one burn rule and differ only in the signal that triggers it. A frozen selection-and-evaluation protocol separates threshold selection from reporting.

**What the project does not claim.** No formal safety guarantee; no new theory (the decision-theoretic results are applications of known ones); no fuel claim from real data; results are from synthetic straight-line-geometry scenarios.

## 2. Status

| Component | Status | Evidence |
|---|---|---|
| M1 orbital mechanics (TLE parsing, SGP4 wrapper, HCW) | done | 5 Vallado reference-state tests (`tests/physics_validation`), plus unit tests |
| M2 sparse graph and false-negative gate | done | 0 misses out of 7 baseline-flagged conjunctions over 5 seeds on one tight synthetic shell (`run_false_negative_gate.py`, run in the audit); no run on a real catalog found in the repository |
| Tracking (Kalman filter, tracked environment) | done | 19 tests; NEES recorded (§8.1) |
| Event labels and analytic forecaster | done | 12 tests; Brier skill recorded (§8.2) |
| Trigger ladder, evaluation harness | done | tests plus recorded runs |
| Frozen protocol, final stream 7 | done, run once | `frozen_thresholds.json`, `results/final_stream7_report.*` |
| Calibration–regret analysis | partial | global logit-affine distortions only; high-L rows clamped (§8.5) |
| ESA CDM retrospective study | done (author's machine) | `results/esa_*`; data not redistributed, so not reproducible from this repo |
| M3 forecasting baselines (LSTM, static GNN, TGN-lite) | partial | scripts and tests exist; a 3–6-scenario pilot, no ranking supported; not the planning forecaster |
| M4 MARL (single-agent, IPPO, MAPPO, MADDPG, constrained MAPPO) | partial | secondary arm, negative result (§8.6) |
| M5 safety filter (HOCBF, multi-threat) | partial | implemented with unit tests; **not audited**, no safety claim |
| Space-Track client | partial | offline tests only; never run live; used in no result |
| CI / experiment tracking (M7) | pending | no CI configuration in the repository |
| Learned CDM-sequence forecaster, noise-mismatch, matched homogeneous-σ control, F400/F200 evaluation | pending | listed as future work (§10) |

## 3. Methodology

### 3.1 Scenarios (`constellai/evaluation/scenarios.py`, `ScenarioConfig`)

3 agents; dt = 10 s; collision radius 50 m; max thrust 0.01 m/s²; up to 200 steps; 15% of agents on a zero-thrust collision course (miss ≤ 0.8·50 m), others pass at 1.5–4 collision radii; approach speed 0.3–1.0 m/s; warning time to closest approach: preset `short` = 150–600 s (main), `long` = 500–1500 s (`warning_config`). Δv is `Σ‖thrust‖·dt` in m/s, per agent, averaged over all agents including unthreatened ones.

Carried over from the 10 Nov README (not re-derived in the audit): linear HCW motion in the LVLH frame about a circular reference orbit at 500 km (WGS-72 constants); the primary threat is a static object at the frame origin, treated as a pseudo-threat with relative state −x_ego; the agents share one policy; collision means separation < r_c on the step grid (strict inequality), a **proxy sphere, not a hard-body radius**. Fuel-scale sanity check: a 50 m lateral shift needs Δv ≈ 2d/T ≈ 0.2–0.7 m/s per threatened agent (straight-line, ignoring HCW coupling); with 15% of agents threatened, the oracle's 0.05 m/s per agent is about 0.3 m/s per threatened agent, consistent with that estimate.

### 3.2 Tracking and forecasting

- `tracking/kalman.py`: `hcw_transition`, `PairFilter` (one filter per ego–threat pair, position-only measurements, isotropic σ_p).
- `tracking/tracker.py`: `TrackedMultiAgentEnv` replaces exact-state observations with filtered tracks. The main experiments **assume cooperative thrust broadcast** between agents (§8.1).
- `forecasting/events.py`: `FirstEntryEngine` and `cumulative_labels` (ground-truth events), `AnalyticForecaster` (samples the filter posterior N(μ, P), propagates with HCW, reports first-passage frequencies with common random numbers; optional σ-gate). **Event definition:** `E_H` = the ego–threat separation falls below r_c at some step within H seconds with both bodies at zero thrust from the forecast time. It is *risk if no action is taken*, not the probability of collision after the planner maneuvers; events are nested, so forecasts are monotone in H by construction. Horizons `HORIZONS_S = (100, 200, 400, 600)` s; horizon index 3 = 600 s, 2 = 400 s, 1 = 200 s.
- **Heterogeneous track quality:** `--sigma-choices 5 50` draws per-pair σ_p from {5, 50} m. In outputs this mixture is labelled `sigma_p = 25`.

### 3.3 Trigger ladder (`evaluation/triggers.py`, `evaluation/heuristic.py`)

All arms burn at full thrust along `away_direction`, and stop when the trigger clears (no hysteresis).

| Arm | Class | Trigger | Policy-name pattern |
|---|---|---|---|
| A0 | `DistanceTriggerPolicy` | current separation < D·r_c | `dist_D*` |
| E | `ThresholdHeuristicPolicy` | straight-line miss < τ·r_c | `heuristic_tau*` |
| E_hcw | `HcwMissTriggerPolicy` | exact-HCW miss of the track mean < τ·r_c | `hcw_tau*` |
| Ehcw_k | `HcwMarginTriggerPolicy` | HCW miss with a k·σ margin | `hcwm_k*` |
| F600 / F400 / F200 | `ProbabilityTriggerPolicy` | forecast P(entry within H) ≥ p* | `prob_analytic_p*_h{3,2,1}` |
| B | `ProbabilityTriggerPolicy` (oracle) | exact event indicator | `prob_oracle_p0.5_h3` |

Threshold grids (`run_ladder.py`, `FAMILIES`): A0 D ∈ {2, 4, 8, 16} (+ A02 {2.5, 3, 3.5}); E τ ∈ {1.2, 1.5, 2, 3, 5} (+ E2 {0.9, 1.0, 1.05, 1.1}); E3 (E_hcw) τ ∈ {0.9, 1.0, 1.05, 1.1, 1.2, 1.5, 2}; HM (Ehcw_k) k ∈ {0, 0.5, 1, 1.5, 2, 3, 4}; F (h3) p* ∈ {0.02, 0.05, 0.1, 0.2, 0.4} + F2 {0.03, 0.07, 0.15, 0.3}; FH2 (h2) and FH1 (h1) p* ∈ {0.03, 0.07, 0.15, 0.3, 0.5}; FC (h3, low thresholds) {0.001 … 0.5}; B p* = 0.5.

### 3.4 Protocol (`evaluation/scenarios.py`, stream constants)

Stream 1 policy training; 2 legacy reference policies (already used); 3 forecaster training; **4 validation**; 5 calibration; **6 development/reporting**; **7 final**. Per family, the threshold with minimum validation Δv subject to validation collision-free ≥ 0.98 is chosen (`select_and_compare.choose`, requires a matching stream-6 run). Reporting uses Wilson intervals for rates (`evaluation/stats.py`) and a **paired bootstrap over scenarios** (trial *i* is the same scenario and noise realisation for every policy). The bootstrap holds the selected thresholds fixed, so it understates selection noise. The final evaluation (`freeze_final.py --execute`) refuses to run unless `frozen_thresholds.json` is committed and clean, and refuses a second run.

### 3.5 Decision-theoretic framing

One-shot cost model: a burn costs c_b, an unprevented event costs L, p* = c_b/L. P1 Bayes ordering; P2 invariance of the threshold frontier under global monotone recalibration; P3 non-invariance under context-dependent calibration; P4 threshold-local regret; P5 value of waiting. These are applications of known results; P4 and the citations are still to be checked (`theory.tex`, kept outside this repository so far).

## 4. Architecture and repository structure

```
constellai/
  common/constants.py            physical constants
  orbital_mechanics/             tle, propagation (SGP4 wrapper), relative_motion (HCW), conjunction,
                                 regime, synthetic, fetch (CelesTrak), spacetrack (Space-Track client)
  graph/                         filters (coarse altitude-band), screening, build (build_graph), validation (false-negative gate)
  simulation/baseline.py         exhaustive non-ML conjunction screening (run_baseline)
  tracking/                      kalman.py (PairFilter), tracker.py (TrackedMultiAgentEnv)
  forecasting/events.py          FirstEntryEngine, AnalyticForecaster, labels
  evaluation/                    scenarios, trials (run_trial, run_trials, summarize_trials), triggers, heuristic,
                                 scaled_env, envs (make_env), policies, stats, monte_carlo, gain_sweep, tiers, nonlinear_env
  models/marl/                   environment, multi_agent_environment, objective (PIDLagrange), constrained_mappo, mappo,
                                 ippo, maddpg, single_agent_baseline, checkpoint, joint_obs, forecast_augmented_environment
  models/tgnn/                   tgn_lite, lstm_baseline, gnn_baseline, dataset, dynamic_graph, graph_dataset, risk_adapter
  safety/                        cbf_filter, multi_threat_filter, filtered_environment
  scripts/                       experiment and analysis entry points (§6)
tests/unit/  tests/physics_validation/
results/                         paper-facing tables and logs (tracked)
outputs/                         m4, m4_scaled, m6, demo, visualizations (tracked); ladder and stream 7 raw outputs are NOT tracked
visualization/generate_figures.py
frozen_thresholds.json           thresholds frozen before the stream 7 run
```

Key scripts: `run_ladder.py` (resumable ladder runner, `done.txt` per job), `evaluate_tracked.py` (evaluates reference and trigger policies; `--val` stream 4, default stream 6, `--final` stream 7), `select_and_compare.py`, `freeze_final.py`, `report_final.py`, `make_figures.py`, `make_final_figure.py`, `analyze_calibration_regret.py`, `profile_esa_cdm.py`, `esa_study.py`, `esa_followup.py`, `make_esa_figures.py`, `reference_baselines.py`, `check_track_consistency.py`, `check_forecast_calibration.py`, `train_tracked.py`, `run_pilot.py`, `analyze_pilot.py` (RL arms only), `run_false_negative_gate.py`, plus M3 training scripts.

## 5. Data

- **Synthetic scenarios:** generated on the fly (`sample_scenario`, deterministic per `(stream, seed, index)`). No external data.
- **TLE ingestion:** `fetch_celestrak_*` (live CelesTrak, tests marked `network`); `fetch_spacetrack_*` (needs `SPACETRACK_USERNAME` and `SPACETRACK_PASSWORD`; never run live).
- **ESA Collision Avoidance Challenge (Kelvins) CDM data:** not redistributed (`data/` is gitignored). Expected layout used by the scripts: `data/esa_cdm/train/train_data.csv` and `data/esa_cdm/Collision Avoidance Challenge - Dataset/kelvins_competition_data/{test_data.csv, test_data_private.csv}`. Profile (author's machine): train 13,154 events / 162,634 CDMs; test 2,167 events. The dataset text defines the target as the final CDM's risk; the high-risk threshold log10 risk > −6 is the competition convention **[unverified: not in the dataset text]**.

## 6. Installation and commands

Python ≥ 3.11 (author: 3.13 on Windows; audit: 3.12.3 on Linux). Dependencies: `sgp4`, `numpy`, `torch`, `scipy`, `matplotlib`; extras `dev` (pytest, pytest-cov) and `esa` (pandas, scikit-learn).

```powershell
git clone https://github.com/Sharvari226/constellAI
cd constellAI
git switch feat/noisy-tracks
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,esa]"
$env:PYTHONPATH = (Get-Location).Path
```

(bash: `source .venv/bin/activate; export PYTHONPATH=$PWD`.)

```powershell
# tests (add -m "not network" offline)
python -m pytest -q -m "not network"

# trigger ladder, heterogeneous mixture, n = 100 per policy (job names match outputs/ladder_het;
# the exact original flags, e.g. --mc-gate, were not recorded)
python constellai/scripts/run_ladder.py --sigmas 25 --sigma-choices 5 50 --families A0 E E3 HM F F2 B --trials 100 --preset short --workers 4 --out-root outputs/ladder_het

# select on stream 4, report on stream 6
python constellai/scripts/select_and_compare.py --root outputs/ladder_het --target 0.98 --boot 2000
python constellai/scripts/make_figures.py --root outputs/ladder_het --sigma 25

# freeze (dry run, then write and COMMIT), then the one-time stream 7 run and report
python constellai/scripts/freeze_final.py --root outputs/ladder_het --trials 300 --expect-jobs 14
python constellai/scripts/freeze_final.py --root outputs/ladder_het --trials 300 --expect-jobs 14 --write
git add frozen_thresholds.json; git commit -m "Freeze thresholds"
python constellai/scripts/freeze_final.py --root outputs/ladder_het --trials 300 --expect-jobs 14 --mc-gate 6 --execute
python constellai/scripts/report_final.py
python constellai/scripts/make_final_figure.py

# diagnostics behind §8.1, §8.2 and §8.10 (commands from the 10 Nov README; not rerun in the audit)
python constellai/scripts/check_track_consistency.py --coop --accel-std 0.0
python constellai/scripts/check_forecast_calibration.py
python constellai/scripts/reference_baselines.py

# calibration-regret analysis
python constellai/scripts/analyze_calibration_regret.py --root outputs/ladder_het --sigmas 25 --nominal cost
python constellai/scripts/analyze_calibration_regret.py --root outputs/ladder_het --sigmas 25 --nominal opt

# ESA study (needs the data under data/esa_cdm)
python constellai/scripts/profile_esa_cdm.py --train <train_data.csv> --test <test_data.csv> --private <test_data_private.csv>
python constellai/scripts/esa_study.py    --train <train_data.csv> --test <test_data.csv> --private <test_data_private.csv>
python constellai/scripts/esa_followup.py --train <train_data.csv> --test <test_data.csv> --private <test_data_private.csv>
python constellai/scripts/make_esa_figures.py
```

Do not run `--execute` again: stream 7 is spent, and any rerun must be labelled post hoc.

## 7. Tests

Audit run (Linux, Python 3.12.3, numpy 2.5.4, scipy 1.18.1, torch 2.15.0, pytest 9.2.0, sandbox without network): **313 passed, 4 deselected (`network` marker)** in 304 s, with `pytest-cov` total **86%** for the `constellai` package (3,067 statements, 423 missed). Per-package coverage: common 100%, forecasting 100%, simulation 100%, tracking 99%, safety 99%, graph 97%, evaluation 96%, models 83%, orbital_mechanics 81%, scripts 59% (how many script files the report includes is **[unverified]**). A first run with the network tests included gave 305 passed, 3 failed (live CelesTrak fetch, no DNS in the sandbox), 1 skipped. On the author's machine (Windows, Python 3.13, network available) the suite gave **308 passed, 1 skipped** before 8 tests were added in the audit.

Tests per file (offline): physics_validation/vallado_reference 5; unit: agent_collision 4, baseline 5, build 4, calibration 3, cbf_filter 6, checkpoint 7, conjunction 5, constrained_mappo 5, env_exposure 5, environment 9, envs 11, filtered_environment 3, filters 4, forecast_augmented_environment 4, forecast_events 12, gain_sweep 13, heuristic 6, ippo 5, kalman 8, maddpg 6, mappo 5, merge_results 2, monte_carlo 7, multi_agent_environment 6, multi_threat_filter 6, network_hooks 7, nonlinear_env 14, objective 8, objective_env 10, policies 16, regime 4, relative_motion 3, scaled_env 9, scenarios 7, screening 3, select_and_freeze 8, single_agent_baseline 5, spacetrack 3, stats 5, synthetic 4, tgn_memory_snapshot 2, tiers 5, tle 9, tracked_env 11, train_m4_ladder 8, trials 5, triggers 7, validation 4.

End-to-end checks run in the audit (not research results): a repeated 10-trial evaluation gave identical per-trial results; E `heuristic_tau1.2`, stream 6, n = 100, gave 0.970 [0.915, 0.990] and Δv 0.234, the same as the author's recorded Windows run; a miniature ladder (n = 10, streams 4 and 6 only) ran through selection, figures, the freeze dry run and the regret script. Stream 7 was never used in the audit.

## 8. Results

All Δv in m/s per agent. Sources: `results/` (tracked) and outputs that are not tracked (stated per table).

### 8.1 Tracking validity (NEES, target 6; 30 episodes, short preset, stream 4; recorded 10 Nov, not rerun)

| Configuration | Primary pair | Neighbour pair |
|---|---|---|
| neighbour thrust **not** broadcast, zero thrust | 5.95–6.09 | 3.34 / 3.49 / 3.60 (σ_p = 5 / 25 / 50) |
| not broadcast, random thrust | same | 6.06 / 6.03 / 6.00 |
| not broadcast, **constant burn** | same | **35.6 / 75.6 / 101.3** |
| **thrust broadcast**, all three policies | 5.95–6.09 | **5.69–5.86** |

The non-cooperative filter is biased when every agent burns the same thrust and cannot be repaired by tuning process noise, so the main experiment **assumes cooperative thrust broadcast** and must say so. The filter shares the simulator's dynamics, so consistency holds by construction; the noise-mismatch experiment (cut) is what would test robustness.

### 8.2 Analytic forecast skill (stream 4, 20 episodes; recorded 10 Nov, not rerun)

Brier skill vs base rate at H = 100 / 200 / 400 / 600 s: σ_p = 5 m 0.996 / 0.970 / 0.896 / 0.790; 25 m 0.844 / 0.733 / 0.538 / 0.440; 50 m 0.584 / 0.483 / 0.319 / 0.257. ECE ≤ 0.015; 20 episodes make the calibration reading suggestive only.

### 8.3 Homogeneous ladder (σ_p ∈ {5, 25, 50} m, short warning, n = 100; `outputs/ladder1`, raw files not tracked; recorded 10 Nov)

Δv in m/s per agent; "val" = stream 4 (threshold chosen), "dev" = stream 6 (reported).

**σ_p = 5 m**

| Arm | Chosen | val free / Δv | dev collision-free | dev Δv |
|---|---|---|---|---|
| A0 | D = 2 | 0.980 / 0.469 | 0.990 [0.946, 0.998] | 0.445 [0.380, 0.513] |
| E | τ = 1.05 | 1.000 / 0.119 | 1.000 [0.963, 1.000] | 0.110 [0.092, 0.129] |
| E_hcw | τ = 1.0 | 1.000 / 0.091 | 0.990 [0.946, 0.998] | 0.082 [0.068, 0.095] |
| F | p* = 0.4 | 1.000 / 0.060 | 1.000 [0.963, 1.000] | 0.058 [0.045, 0.073] |
| B | oracle | 1.000 / 0.046 | 1.000 [0.963, 1.000] | 0.047 [0.033, 0.061] |

**σ_p = 25 m**

| Arm | Chosen | val free / Δv | dev collision-free | dev Δv |
|---|---|---|---|---|
| A0 | D = 2.5 | 1.000 / 0.629 | 1.000 [0.963, 1.000] | 0.633 [0.559, 0.708] |
| E | τ = 1.2 | 0.990 / 0.246 | 0.970 [0.915, 0.990] | 0.235 [0.204, 0.267] |
| E_hcw | τ = 1.2 | 0.990 / 0.215 | 0.970 [0.915, 0.990] | 0.211 [0.184, 0.238] |
| F | p* = 0.3 | 0.990 / 0.107 | **0.940** [0.875, 0.972] | 0.112 [0.091, 0.133] |
| B | oracle | 1.000 / 0.049 | 1.000 [0.963, 1.000] | 0.056 [0.037, 0.080] |

**σ_p = 50 m**

| Arm | Chosen | val free / Δv | dev collision-free | dev Δv |
|---|---|---|---|---|
| A0 | D = 2.5 | 0.980 / 0.623 | 0.990 [0.946, 0.998] | 0.647 [0.575, 0.721] |
| E | τ = 1.5 | 0.980 / 0.363 | 0.950 [0.888, 0.978] | 0.427 [0.383, 0.473] |
| E_hcw | τ = 1.5 | 0.980 / 0.368 | 0.980 [0.930, 0.994] | 0.406 [0.362, 0.453] |
| F | p* = 0.07 | 0.980 / 0.395 | 0.990 [0.946, 0.998] | 0.423 [0.381, 0.465] |
| B | oracle | 1.000 / 0.053 | 1.000 [0.963, 1.000] | 0.051 [0.037, 0.065] |

Paired differences on the development stream (first minus second; positive Δv = first costs more; negative safety = first is less safe):

| σ_p | A0 − E | E − E_hcw | E_hcw − F | F − B |
|---|---|---|---|---|
| 5 m, Δv | +0.336 [0.276, 0.395] | +0.028 [0.014, 0.043] | +0.023 [0.015, 0.032] | +0.012 [0.003, 0.021] |
| 5 m, safety | −0.010 [−0.030, 0.000] | +0.010 [0.000, 0.030] | −0.010 [−0.030, 0.000] | 0.000 |
| 25 m, Δv | +0.398 [0.333, 0.462] | +0.024 [0.006, 0.043] | +0.099 [0.078, 0.121] | +0.056 [0.033, 0.077] |
| 25 m, safety | +0.030 [0.000, 0.070] | 0.000 | +0.030 [−0.010, 0.080] | −0.060 [−0.110, −0.020] |
| 50 m, Δv | +0.220 [0.161, 0.279] | +0.021 [−0.013, 0.055] | −0.017 [−0.049, 0.015] | +0.373 [0.337, 0.410] |
| 50 m, safety | +0.040 [0.010, 0.080] | −0.030 [−0.070, 0.000] | −0.010 [−0.030, 0.000] | −0.010 [−0.030, 0.000] |

Reading: (1) extrapolating the track is worth a lot: A0 costs about 4× E at σ_p = 5 m, 2.7× at 25 m and 1.5× at 50 m (a coarser A0 grid had inflated this to "3–10×"). (2) Correct dynamics save 0.02–0.03 m/s, i.e. 25%, 10% and 5% of E's cost at 5 / 25 / 50 m; the interval excludes 0 at 5 and 25 m. (3) F vs. a point forecast with correct dynamics: cheaper at 5 m (28% of E_hcw's cost), no demonstrated difference at 50 m, and at 25 m F is 3 points less safe on the development stream (0.94 vs 0.97), so the comparison is **not matched**. (4) F − B is the value of perfect state knowledge, not forecasting headroom; it grows with noise (0.012 → 0.056 → 0.373) because tracking error limits any track-conditioned rule. (5) Selection noise is visible: thresholds chosen on 100 validation trials regress on the development stream (F at 25 m: 0.99 → 0.94; E at 50 m: 0.98 → 0.95).

### 8.4 Heterogeneous mixture {5, 50} m

Selection (n = 100; `results/table_selection_sigma25.csv`): thresholds chosen on stream 4, reported on stream 6.

| Arm | Policy | Val free / Δv | Stream 6 collision-free [95% CI] | Stream 6 Δv [95% CI] |
|---|---|---|---|---|
| A0 | dist_D4 | 1.000 / 1.140 | 1.000 [0.963, 1.000] | 1.124 [1.070, 1.181] |
| E | heuristic_tau1.2 | 0.980 / 0.199 | 0.970 [0.915, 0.990] | 0.234 [0.202, 0.267] |
| E_hcw | hcw_tau1.5 | 1.000 / 0.272 | 0.980 [0.930, 0.994] | 0.280 [0.249, 0.311] |
| Ehcw_k | hcwm_k1 | 0.990 / 0.220 | 1.000 [0.963, 1.000] | 0.248 [0.218, 0.280] |
| F600 | prob_analytic_p0.15_h3 | 0.980 / 0.171 | 1.000 [0.963, 1.000] | 0.180 [0.152, 0.208] |
| B | prob_oracle_p0.5_h3 | 1.000 / 0.047 | 1.000 [0.963, 1.000] | 0.053 [0.038, 0.069] |

**Final evaluation, stream 7, run once** (n = 300 fresh scenarios, thresholds frozen in commit `8618a56`, `--mc-gate 6`; `results/final_stream7_report.*`):

| Arm | Collision-free [95% CI] | Δv [95% CI] |
|---|---|---|
| A0 | 1.000 [0.987, 1.000] | 1.150 [1.116, 1.187] |
| E | **0.953 [0.923, 0.972]** | 0.218 [0.201, 0.235] |
| E_hcw | 0.990 [0.971, 0.997] | 0.279 [0.260, 0.297] |
| Ehcw_k | 0.990 [0.971, 0.997] | 0.242 [0.224, 0.260] |
| F600 | 0.990 [0.971, 0.997] | 0.183 [0.167, 0.200] |
| B | 1.000 [0.987, 1.000] | 0.047 [0.040, 0.055] |

Paired differences on stream 7 (first minus second): A0 − E +0.932 [0.899, 0.967]; E − E_hcw −0.061 [−0.076, −0.046] (E is less safe: −0.037 [−0.060, −0.017]); E_hcw − Ehcw_k +0.036 [0.027, 0.046] at identical safety; **Ehcw_k − F600 +0.059 [0.047, 0.071] at equal safety (0.990 both)**; E_hcw − F600 +0.096 [0.083, 0.108]; F600 − B +0.136 [0.122, 0.150].

Against the pre-stated predictions: fixed-τ loses to margin (supported at matched safety); margin ≈ F (not supported: F600 is cheaper); F ≤ margin (supported).

### 8.5 Calibration–regret analysis (`results/regret_het_*.txt`, n = 100, mixture, F600 family)

With p* = c_b/L (c_b = 0.33 m/s, an estimate): L = 3 (p* = 0.11, tuned optimum 0.2) regret of the nominally calibrated threshold +0.058 [0.023, 0.088]; L = 10 (p* = 0.033, tuned optimum 0.1) +0.221 [0.196, 0.250]. Rows marked "clamped" (all L ≥ 30 under the cost nominal) are lower bounds at the grid edge, and the L ≥ 30 rows under the tuned nominal have intervals up to +2.2; neither is interpretable. Global logit-affine distortions only.

### 8.6 RL arm (secondary; negative result; recorded 10 Nov; `outputs/pilot1`, not tracked)

Critic diagnosis (single seed, in-sample explained variance): multiplying mission rewards by 0.05 raised reward-critic explained variance from about 0.00 to 0.85 (last 10 episodes) with the same code; ruled out as causes were observability of the slot error, the placeholder mission term and the fuel penalty (removing it gave 0.67). Cost-critic EV is about 0.64 with tracked observations.

Policy pilot (short preset, σ_p = 25 m, budget 5.7, 3 seeds × 300 episodes, n = 100, stream 6):

| Policy | Collision-free (mean ± sd over seeds) | Δv / agent | Saturation |
|---|---|---|---|
| RL reactive | 0.913 ± 0.038 | 19.75 ± 3.24 | 0.544 |
| RL + oracle forecast | 0.910 ± 0.044 | 23.68 ± 2.75 | 0.679 |
| best rule (E_hcw / F, §8.3) | 0.97–0.99 | 0.1–0.4 | n/a |

Paired RL reactive − oracle: collision-free +0.003 [−0.073, +0.080]; Δv −3.9 [−7.9, 0.0]. The learned planner burns roughly **50–180×** the fuel of the rules (19.75 / 0.4 ≈ 50; 19.75 / 0.11 ≈ 180; the 10 Nov README's "about 50×" is the conservative end), and the oracle forecast gave it no benefit at this scale. The PID constraint is met (observed discounted cost 1.5–4.6 against a budget of 5.7) but is not binding and fuel is not priced, so the policy burns continuously. In seed 0, 90% of steps have a mean action beyond the thrust limit that the environment clips; seed 1 stays mostly inside the limit and still burns about 19 m/s. To test before any RL claim: bounded actions (tanh mean), fuel as the optimised quantity with risk as the constraint, longer training.

### 8.7 ESA CDM retrospective study (`results/esa_followup.txt`, `results/regret_by_cut_signal_ratio.csv`, `results/context_by_covariance_tertile.csv`)

Test events/positives at the 2/3/5-day cuts: 2,167/150, 2,045/138, 1,753/106; prevalence 2.7% (train) vs 6.9% (test). Cost model: a flag costs r, a miss costs 1; **no fuel claim**. Tuned threshold on ESA's own risk has lower regret than a tuned gradient-boosting model in 8 of 9 cut × r cells (e.g. 2 d, r = 0.05: −0.0107 [−0.0152, −0.0067]); the learned model adds nothing. S4 calibration ratio (mean p / prevalence) by chaser-covariance tertile: validation 0.83 / 0.89 / 1.08 (2 d); test prior-shift-corrected (diagnostic) 0.54 / 0.58 / 0.89 (2 d), 0.45 / 0.44 / 1.04 (3 d), 0.59 / 0.64 / 1.40 (5 d). Regret does not track that gradient consistently, so "which calibration measure predicts regret" is **inconclusive**. The ESA `risk` feature's untuned rows are artefacts (10^risk ≈ 0 flags nothing).

### 8.8 Figure and table locations

`results/`: selection and paired tables (`table_*_sigma25.{csv,tex}`), stream 7 report, regret and ESA CSV/TXT. Generated by the scripts and not tracked: `outputs/figures/ladder_het/` (frontier, cost bars), `outputs/figures/final_stream7/`, `outputs/esa_study/figures/`. Tracked earlier figures: `outputs/visualizations/`, `outputs/demo/` (M1–M3 demonstrations; not re-checked in this audit).

### 8.9 Hypotheses and claims register (carried over from the 10 Nov README, updated with stream 7 and ESA)

| # | Hypothesis | Status |
|---|---|---|
| H1 | Forecast-type triggers beat a purely reactive trigger on the Δv-vs-collision frontier. | **Supported.** E beats A0 with intervals excluding 0 at every tested noise level (pilot); in the mixture on stream 7, A0 − E = +0.932 [0.899, 0.967]. |
| H1b | Probabilistic forecasting beats a point forecast with correct dynamics. | **Mixed.** Homogeneous pilot: cheaper at σ_p = 5 m, no difference at 50 m, unmatched safety at 25 m. Mixture, stream 7: F600 cheaper than E_hcw by 0.096 [0.083, 0.108] at equal safety; one condition, thresholds from an n = 100 selection. |
| H2 | Calibration matters for planning. | **Partly addressed.** A tuned threshold is invariant to global monotone recalibration (P2). The untuned cost-ratio threshold has positive regret (§8.5, n = 100, lower bounds at high L). ESA retrospective study: inconclusive (§8.7). |
| H3 | A learned forecaster adds value over physics-based propagation. | **Untested.** Synthetic learned-forecaster arms were cut; on ESA data a gradient-boosting model on last-CDM features did no better than a tuned threshold on ESA's own risk. |
| H4 | Multi-horizon information improves the trade-off. | **Untested at the frozen protocol.** F400 and F200 were not part of the frozen set. |

Claims the evidence supports (pilot-level unless stated):

| # | Claim | Evidence |
|---|---|---|
| C1 | Forecast-type triggers reach ≥ 0.97–1.00 collision-free at 0.06–0.4 m/s per agent on short-warning synthetic scenarios, where a current-distance trigger needs 0.45–0.65 m/s. | §8.3, §8.4 |
| C2 | Extrapolating the track (E) beats a reactive trigger (A0); intervals exclude 0. | §8.3, §8.4 |
| C3 | Exact dynamics save 0.02–0.03 m/s in the **homogeneous** setting at σ_p = 5 and 25 m; **not reproduced** in the {5, 50} mixture. | §8.3 vs §8.4 |
| C4 | At σ_p = 5 m a probabilistic trigger is cheaper than a point trigger with exact dynamics (0.023 m/s). | §8.3 |
| C5 | In the {5, 50} mixture on stream 7 (n = 300), F600 is cheaper than the k·σ margin by 0.059 [0.047, 0.071] at equal safety, and fixed-τ HCW loses to the margin (+0.036 [0.027, 0.046]); one condition, thresholds from 100 validation trials. | §8.4 |
| C6 | Perfect state knowledge would cut Δv to about 0.05 m/s at every noise level; the gap to the best track-conditioned rule grows with noise. | §8.3, §8.4 |
| C7 | The filter is consistent under cooperative thrust broadcast and inconsistent without it under correlated burns. | §8.1 |
| C8 | The analytic forecast has positive Brier skill at all horizons and noise levels. | §8.2 |
| C9 | The RL planner is far less fuel-efficient than the rules and did not benefit from an oracle forecast. | §8.6 |

Do **not** write: that calibration improves planning or that a calibration measure predicts Δv or regret on real data (ESA inconclusive); that heterogeneity explains F's advantage (no matched control); that multi-horizon forecasting helps (not evaluated); that a learned forecaster beats physics propagation; that E beats E_hcw (E missed the safety target); that F has headroom beyond the track-conditioned rules (F − B measures tracking, not forecasting); any safety guarantee, other constellation size, multiple simultaneous threats, maneuvering threats or real-orbit result; any fuel claim from real data; any ranking of M3 models.

### 8.10 Reference policies (recorded 10 Nov, stream 6, 100 episodes, γ = 0.99; not rerun in the audit)

| Preset | Policy | Collision-free [95% CI] | Agent collision rate | Δv / agent | Discounted shaped cost, mean (p90) |
|---|---|---|---|---|---|
| short | zero thrust | 0.550 [0.452, 0.644] | 0.197 | 0 | 14.3 (23.4) |
| short | random thrust | 0.730 [0.636, 0.807] | 0.107 | 17.5 | 11.0 (21.9) |
| short | full burn | 0.770 [0.678, 0.842] | 0.133 | 30.8 | 11.4 (23.3) |
| long | zero thrust | 0.550 [0.452, 0.644] | 0.197 | 0 | 9.9 (19.1) |
| long | random thrust | 0.960 [0.902, 0.984] | 0.017 | 19.0 | 3.0 (10.4) |
| long | full burn | 0.860 [0.779, 0.915] | 0.083 | 32.9 | 6.1 (15.1) |

The short preset has headroom (a random-thrust policy is 73% collision-free at 17.5 m/s); the long preset largely does not, which is why short warning is the main experiment. Even 31 m/s of undirected thrust leaves 23% of short trials with a collision.

### 8.11 M3 forecasting pilot (recorded 10 Nov; an older task and a different event)

M3 predicts "separation below 300 km within 90 minutes", not the planning event. LSTM and GNN: 6 test scenarios at a 2-minute step; TGN-lite: 3 test scenarios at a 5-minute step; 3 seeds.

| Model | AP (seed range) | Precision | Recall |
|---|---|---|---|
| LSTM | 0.756 (0.66–0.84) | 0.760 | 0.864 |
| Static GNN | 0.183 | 0.162 | 0.773 |
| TGN-lite (point) | 0.246 (0.18–0.31) | 0.047 | 0.711 |
| TGN-lite (Beta head) | 0.014, ECE 0.334 | | |

The effective sample is 3–6 scenarios and the comparison is unmatched, so it supports no ranking. TGN-lite does not read event timestamps, so continuous-time claims are unsupported; it is not the forecaster for the central experiment.

### 8.12 Established methods vs. intended contribution

HCW dynamics, Kalman filtering, PPO/MAPPO, Lagrangian constrained RL, the PID multiplier, CBF filters and TGN/LSTM/GNN risk classifiers are established. Threshold rules on miss distance and risk probability are standard operational practice in some form. The trigger ladder with a shared burn rule, the exact-dynamics control and the validation-then-frozen protocol are our experimental design. The decomposition of the value of forecasting into extrapolation, dynamics fidelity, probabilistic handling and perfect state knowledge is the intended contribution; **novelty is not established.**

## 9. Limitations and assumptions

- **Final thresholds were selected on 100 validation trials.** An n = 300 selection rerun was started and abandoned (1 of 18 jobs after about 9 h on the shared machine); no number from it is used.
- **Stream 7 was run once.** It is spent; retuning on it is not allowed.
- **E missed the 0.98 target** on stream 7 (0.953), so its cost advantage over E_hcw is not a matched comparison.
- **One heterogeneity condition, no matched homogeneous-σ control:** "heterogeneity explains F's advantage" is a hypothesis.
- **F400 and F200 were never in the frozen set;** there is no multi-horizon result. Earlier "infeasible" readings were a misleading script message (fixed in the audit).
- **Cooperative thrust broadcast** is assumed; without it the filter is inconsistent under correlated burns.
- **Straight-line, single-threat synthetic scenarios** favour straight-line rules; the burn rule is full-thrust with no hysteresis.
- **The ESA study is retrospective;** the learned model is gradient boosting on last-CDM features, not a sequence model; the −6 label threshold is a convention.
- **The M5 safety filter is unaudited,** the RL arm is a negative result, and the M3 pilot supports no model ranking.
- **Known issue (from the 10 Nov README, [unverified] in the audit):** `test_mappo_learns_to_reduce_collisions` starts the agents inside the collision radius and is likely degenerate; some docstrings may still be stale. (Tracked `.pyc` files are untracked by the audit patch.)
- **Reproducibility gaps:** raw ladder and stream 7 JSON files are not tracked; the ESA data is not redistributed; the exact original `--mc-gate` for `ladder_het` was not recorded; the audit ran on Linux/Python 3.12 while results were produced on Windows/Python 3.13 (one n = 100 policy reproduced exactly).

## 10. Future scope

Matched homogeneous-σ control; F400/F200 at the frozen protocol and n ≥ 300 selection; FC low-threshold family to unclamp the regret rows; learned CDM-sequence forecaster on ESA data; context-dependent calibration experiments; noise-mismatch; harder scenario families (simultaneous threats, maneuvering threat); RL repair (bounded actions, fuel as objective); M5 audit; Space-Track history; CI.

## 11. References (as used in the project notes; **[unverified]**: not re-checked against the sources)

Vallado, *Revisiting Spacetrack Report #3* (reference satellite #00005, used by `tests/physics_validation`); Markov Decision Processes for Satellite Maneuver Planning and Collision Avoidance (arXiv:2501.02667); Greco et al., robust Bayesian agent for collision avoidance manoeuvre planning (Strathclyde, 2021); ConstrainedZero (IJCAI 2024); Stooke et al., PID Lagrangian methods (ICML 2020); ESA Kelvins Collision Avoidance Challenge dataset (Zenodo) and Uriot et al., arXiv:2008.03069; decision-theory priors for the theory section (Schervish; Rothblum–Yona; Hu–Wu, FOCS 2024; Gopalan et al.; Kleinberg et al.; Zhao et al.) to be checked before citation. Sources suggested by the team's reviewer (a NASA conjunction-assessment AI/ML compendium, a 2025 high-covariance conjunction-prediction paper, a 2026 belief-space planning paper) could **not** be confirmed and must not be cited until located. The related-work notes above were read at abstract level only.

## 12. Team (as recorded in the earlier README; names and final attribution to be confirmed by the team)

| Role | Focus |
|---|---|
| Dev A + Dev B | M1 (orbital mechanics) + M6 (simulation and evaluation) |
| Dev C | M2 (sparse graph) + M3 (temporal forecasting) |
| Dev D | M4 (constrained MARL), solo |
| Mentor + Dev D | M5 (safety filter), paired |

The trigger-ladder, tracking, forecaster, protocol, ESA study and final-evaluation work belongs to the project lead and is not attributed here per author **[to be filled by the team]**.

## 13. Working conventions

- Work on feature branches; do not push to `main` directly. Add files by explicit path; never `git add -A`.
- Every experiment states what is tested, the controlled variables, the metrics, what would support or falsify the hypothesis, and what remains unjustified.
- Results contradicting the hypothesis are reported as such; results from before the environment fix are never mixed into a table.
- Status claims are re-verified against code, tests and logs before they are reused in the paper.

Licence: to be added.
