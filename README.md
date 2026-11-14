# ConstellAI

**When does forecast quality reduce the fuel needed for risk-triggered satellite maneuvers, and which calibration measure predicts it?**

> **Status (14 Nov 2026): research prototype. The final evaluation (stream 7, n = 300, thresholds frozen from the n = 100 stream-4 selection) was run once; see §6.5. Everything else is pilot-level.**
> Tags used throughout: **[V]** verified from output seen, **[A]** assumed (not seen), **[P]** pending (not run), **[X]** cut or deferred.
> Development-protocol numbers (n = 100, thresholds chosen on a validation stream, read on a development stream) are for drafting structure and arguments, **not headline claims**. §7 lists what may and may not be written.

Supersedes the 10 Nov README. What changed is in §0.

---

## 0. What changed since the 10 Nov README

| Item | 10 Nov | 14 Nov |
|---|---|---|
| Research question | "Does calibrated multi-horizon forecasting improve the safety–fuel trade-off?" | Reframed: *when* does forecast quality cut Δv, and *which calibration measure* predicts it (handoff 13 Nov). The 10 Nov question stays as motivation. |
| Heterogeneous track quality | not present | Done: `--sigma-choices`, k·σ margin arm (Ehcw_k), `ladder_het` (14 jobs, n = 100) finished [V]. |
| Horizon families (F400, F200) | untested | Implemented; pilot reading in the handoff: F600 ≈ F400, F200 costlier at σ ≥ 25 (not re-seen) [A]. Included in the n = 300 rerun [P]. |
| Calibration / regret analysis | planned (C3) | Done for global monotone distortions (`analyze_calibration_regret.py`) [A: output not re-seen]. |
| Theory | none | P1–P5 as applications of existing results (§8). |
| Real data | Space-Track planned | ESA Kelvins CDMs planned (profile script written, not run); Space-Track deferred. |
| Cut | | Scenario audit, synthetic learned-forecaster arms, noise-mismatch, RL repair, M5, scaling. Move to future work. |
| Final stream | stream 7 | unchanged: stream 7, run once, frozen thresholds. |
| Tests | 299 passed, 1 skipped | **308 passed, 1 skipped** [V]. |
| Final stream 7 | not run | **Run once**, n = 300, frozen thresholds (commit `8618a56`) [V]; §6.5. |
| n = 300 selection rerun | planned | **Abandoned** (1/18 jobs after ~9 h); thresholds come from the n = 100 selection [V]; §6.4. |
| ESA CDM study | planned | **Run** as a retrospective study; inconclusive on calibration-vs-regret [V]; §6.6. |

Resolved assumptions: Δv is in m/s (Σ‖thrust‖·dt per agent, averaged over all agents) [V from 10 Nov README §2]; `h3` is the 600 s horizon [V: "H = 600 s so far"]. Still assumed: `h1/h2` = 200/400 s and `FH1/FH2` = F200/F400.

---

## 1. Aims and question

Low Earth orbit is congested; conjunction management combines anticipating risk, deciding under uncertainty and conserving fuel.

**Working question (13 Nov):** *When does forecast quality (uncertainty-aware, context-calibrated risk probability) reduce the Δv needed for risk-triggered maneuvers, and which calibration measure predicts it?*

### 1.1 How the evidence shaped the design

A first-pass RL planner burned about 20 m/s per agent while a 40-line threshold rule matched or beat its safety at about 0.1–0.4 m/s. In deterministic dynamics a forecast that is a function of the observed track carries no information a learned policy could not compute, so the meaningful comparison is between **trigger signals**: current distance → straight-line miss → exact-dynamics miss → k·σ margin → calibrated probability → oracle. The ladder is primary; RL is secondary and reported as a negative result.

### 1.2 Hypotheses

| # | Hypothesis | Status |
|---|---|---|
| H1 | Forecast-type triggers beat a purely reactive trigger on the Δv–safety frontier. | **Supported (pilot).** A0 − E > 0, intervals exclude 0 (§6.2). |
| H1b | Probabilistic handling beats a point forecast with correct dynamics. | **Mixed.** Cheaper at σ_p = 5 m (homogeneous); no difference at 50 m; unmatched at 25 m (homogeneous). In the {5, 50} mixture F600 beats the k·σ margin by 0.068 [0.044, 0.092] at equal pilot safety, n = 100, one condition (§6.3). |
| H2 | Calibration matters for planning. | **Theory + analysis exist, empirical test pending.** Tuned thresholds are invariant to global monotone recalibration (P2). Calibration matters for untuned cost-ratio thresholds (regret analysis) and, by P3, when calibration varies with context. Real-data test: ESA study [P]. |
| H3 | A learned forecaster adds value over physics-based propagation. | **Not testable on synthetic data (cut).** Planned only on ESA CDMs [P]. |
| H4 | Multi-horizon information improves the trade-off. | **Pilot reading only:** F600 ≈ F400; F200 costlier at σ ≥ 25 [A]. n = 300 rerun pending. |

### 1.3 Non-claims

No formal safety guarantee. **No novelty claim for theory** (prior art: Schervish representation, Rothblum–Yona, Hu–Wu FOCS'24, Gopalan et al., Kleinberg et al., decision calibration of Zhao et al.). Tracking noise is a synthetic sensitivity setting. Calibration is relative to the scenario distribution. Everything is synthetic, straight-line-geometry scenarios; **no fuel claim from real data**.

---

## 2. Setting and modelling choices

| Item | Choice |
|---|---|
| Dynamics | Linear HCW relative motion (LVLH), circular reference orbit at 500 km (WGS-72) |
| Primary threat | Static object at the frame origin, pseudo-threat with relative state −x_ego |
| Agents | N = 3 controlled satellites, shared policy; thrust limit 0.01 m/s² per axis |
| Step | dt = 10 s, up to 200 steps |
| Collision | Separation < r_c = 50 m on the step grid (proxy sphere, not a hard-body radius) |
| Fuel metric | Δv = Σ‖thrust‖·dt (m/s), per agent, averaged over all agents including unthreatened ones |
| Tracks | Kalman filter per ego–threat pair, position-only measurements, isotropic σ_p ∈ {5, 25, 50} m (0.1, 0.5, 1.0 r_c) |
| **Heterogeneous tracks** | `--sigma-choices 5 50`: per-pair σ_p drawn from {5, 50} m. In outputs this is labelled "sigma_p = 25" (the mixture, not σ = 25). |
| Neighbour thrust | **Cooperative assumption:** neighbours broadcast applied thrust (§6.1) |
| Scenarios | Built backwards from a chosen closest approach; 15% of agents on a zero-thrust collision course (miss ≤ 40 m), others pass at 75–200 m; warning time short = 150–600 s (main) or long = 500–1500 s |

Fuel-scale sanity check: a 50 m lateral shift needs Δv ≈ 2d/T ≈ 0.2–0.7 m/s per threatened agent; with 15% threatened, the oracle's 0.05 m/s per agent is about 0.3 m/s per threatened agent, consistent.

---

## 3. What is implemented

Branch `feat/noisy-tracks`; tag `baseline-pre-objective` = `edcaf04`; HEAD `f60a08b` = `origin/feat/noisy-tracks` [V]; `main` untouched.

| Area | Files | Verification |
|---|---|---|
| Environment fixes | `models/marl/multi_agent_environment.py` | Two-phase `step()` (order-independent collisions/costs); fuel penalty once; nominal-path mission term (legacy `primary_shell` kept) |
| Kalman filter, tracked env | `tracking/kalman.py`, `tracker.py` | Transition matrices match propagator; NEES consistency test with power; information parity across arms |
| Labels, analytic forecaster | `forecasting/events.py` | Labels equal brute-force propagator and the env's first collision step; monotone in horizon; exact in zero-uncertainty limit; σ-gate 2.7× faster, identical decisions |
| Trigger ladder | `evaluation/heuristic.py`, `triggers.py` | Direction rule shared across arms; each trigger tested alone and in the tracked env |
| Heterogeneity, margin policy | `--sigma-choices`, `HcwMarginTriggerPolicy` | 25 tests pass (handoff) |
| Calibration–regret analysis | `scripts/analyze_calibration_regret.py` | Global monotone distortions only |
| Constrained objective (RL arm) | `objective.py`, `constrained_mappo.py`, `joint_obs.py`, `checkpoint.py` | PID Lagrangian, privileged critic |
| Evaluation | `evaluate_tracked.py`, `run_ladder.py`, `select_and_compare.py`, `run_pilot.py` | Raw per-trial results saved; paired bootstrap over scenarios |
| ESA profiler | `scripts/profile_esa_cdm.py` | Written, **not run** [P] |

Suite: **308 passed, 1 skipped (~9.9 min)** [V]. Uncommitted: `evaluation/scaled_env.py` adds `ValueError` guards on non-positive `fuel_budget`, `max_thrust_mps2`, `dt` [V from diff]; cannot change valid-parameter results; currently stashed ("scaled_env validation").

---

## 4. Planners compared

One burn rule for all trigger arms: when triggered, full thrust along −m, where m is the predicted closest-approach vector from track means under straight-line motion (perpendicular to relative velocity if exactly head-on); stop when the trigger clears (no hysteresis). Only the trigger signal differs.

| Arm | Trigger | Policy names |
|---|---|---|
| A0 | current separation < D·r_c | `dist_D*` |
| E | straight-line miss < τ·r_c | `heuristic_tau*` |
| E_hcw | exact-HCW miss of track mean < τ·r_c | `hcw_tau*` |
| Ehcw_k | HCW miss with k·σ margin | `hcwm_k*` |
| F600 / F400 / F200 | analytic-forecast P(entry within H) ≥ p* | `prob_analytic_p*_h{3,2,1}` |
| B | oracle event indicator | `prob_oracle_p0.5_h3` |
| RL | Constrained MAPPO on tracked observations (secondary) | checkpoints |

Runner families: `A0 A02 E E2 E3 HM F F2 FH2 FH1 FC B`. F600 in selection tables pools the `F` and `F2` grids [A].

**Event.** E_H: separation falls below r_c within H seconds with both bodies at zero thrust from forecast time (risk if no action). Horizons {100, 200, 400, 600} s, nested. **Analytic forecaster:** sample the filter posterior N(μ, P), propagate with Φ^s, report first-passage frequencies (common random numbers; deterministic and monotone). **RL objective:** mission reward (deviation from zero-thrust nominal path, scaled ×0.05) subject to a budget on a shaped proximity cost (a proxy, not a bound on collision probability).

---

## 5. Experimental protocol (frozen)

| Stream | Role |
|---|---|
| 1 | RL policy training |
| 2 | Legacy; already used for reference policies and old diagnostics |
| 3 / 5 | Forecaster training / calibration (reserved for ESA-style learned forecaster) |
| **4** | **Validation:** choose each family's threshold: min Δv s.t. validation collision-free ≥ 0.98 |
| **6** | **Development/report:** read selected thresholds, paired bootstrap over scenarios. Looked at repeatedly. |
| **7** | **Final:** one run with frozen, committed thresholds. Never read. |

Intervals: Wilson for rates; paired bootstrap (`--boot 2000`) over scenarios (trial *i* = same scenario and noise for every policy). Bootstrap holds selected thresholds fixed, so it understates selection noise. At n = 100 the ≥ 0.98 constraint allows ≤ 2 failures and cannot be certified statistically. Seed `--noise-seed 12345`. Resumable runners via `done.txt`; use a new `--out-root` when `--trials` changes. `--final` uses stream 7 (help text saying "stream 2" is stale).

---

## 6. Results

### 6.1 Tracking validity (NEES, target 6; 30 episodes, short preset, stream 4)

| Configuration | Primary pair | Neighbour pair |
|---|---|---|
| thrust **not** broadcast, zero thrust | 5.95–6.09 | 3.34 / 3.49 / 3.60 (σ_p = 5/25/50) |
| not broadcast, random thrust | same | 6.06 / 6.03 / 6.00 |
| not broadcast, **constant burn** | same | **35.6 / 75.6 / 101.3** |
| **thrust broadcast**, all policies | 5.95–6.09 | **5.69–5.86** |

The main experiment **assumes cooperative thrust broadcast** and must say so. The filter shares the simulator's dynamics, so consistency holds by construction; the noise-mismatch test is cut [X].

### 6.2 Homogeneous ladder (σ_p ∈ {5, 25, 50}, short warning, n = 100; `outputs/ladder1`)

Δv m/s per agent; "dev" = stream 6.

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

Paired differences on stream 6 (first − second; positive Δv = first costs more):

| σ_p | A0 − E | E − E_hcw | E_hcw − F | F − B |
|---|---|---|---|---|
| 5 m, Δv | +0.336 [0.276, 0.395] | +0.028 [0.014, 0.043] | +0.023 [0.015, 0.032] | +0.012 [0.003, 0.021] |
| 5 m, safety | −0.010 [−0.030, 0.000] | +0.010 [0.000, 0.030] | −0.010 [−0.030, 0.000] | 0.000 |
| 25 m, Δv | +0.398 [0.333, 0.462] | +0.024 [0.006, 0.043] | +0.099 [0.078, 0.121] | +0.056 [0.033, 0.077] |
| 25 m, safety | +0.030 [0.000, 0.070] | 0.000 | +0.030 [−0.010, 0.080] | −0.060 [−0.110, −0.020] |
| 50 m, Δv | +0.220 [0.161, 0.279] | +0.021 [−0.013, 0.055] | −0.017 [−0.049, 0.015] | +0.373 [0.337, 0.410] |
| 50 m, safety | +0.040 [0.010, 0.080] | −0.030 [−0.070, 0.000] | −0.010 [−0.030, 0.000] | −0.010 [−0.030, 0.000] |

**Reading.** (1) Extrapolation is worth a lot: A0 costs about 4×/2.7×/1.5× E at σ_p = 5/25/50. (2) Exact dynamics saves 0.02–0.03 m/s (25%/10%/5% of E's cost), interval excludes 0 at 5 and 25 m. (3) F vs E_hcw: cheaper at 5 m (28% of E_hcw's cost); no demonstrated difference at 50 m; at 25 m **not matched** (F 3 points less safe), so not a win. (4) F − B is the value of perfect state knowledge, not forecasting headroom; it grows with noise (0.012 → 0.056 → 0.373). (5) Selection noise is visible: F at 25 m regresses 0.99 → 0.94 from validation to development.

### 6.3 Heterogeneous track quality, mixture {5, 50} m, n = 100 [V] (`outputs/ladder_het`, 14 jobs, all `done=ok`, no log errors)

Selected on stream 4, reported on stream 6.

| Family | Selected | Val free | Val Δv | Dev collision-free [95% CI] | Dev Δv [95% boot CI] |
|---|---|---|---|---|---|
| A0 | dist_D4 | 1.000 | 1.140 | 1.000 [0.963, 1.000] | 1.124 [1.070, 1.181] |
| E | heuristic_tau1.2 | 0.980 | 0.199 | 0.970 [0.915, 0.990] | 0.234 [0.202, 0.267] |
| E_hcw | hcw_tau1.5 | 1.000 | 0.272 | 0.980 [0.930, 0.994] | 0.280 [0.249, 0.311] |
| Ehcw_k | hcwm_k1 | 0.990 | 0.220 | 1.000 [0.963, 1.000] | 0.248 [0.218, 0.280] |
| F600 | prob_analytic_p0.15_h3 | 0.980 | 0.171 | 1.000 [0.963, 1.000] | 0.180 [0.152, 0.208] |
| F400, F200 | not run in this job set; "no threshold reaches the target" is the empty-family message | | | | |
| B | prob_oracle_p0.5_h3 | 1.000 | 0.047 | 1.000 [0.963, 1.000] | 0.053 [0.038, 0.069] |

| Pair (stream 6) | Δv | Collision-free |
|---|---|---|
| A0 − E | +0.890 [+0.839, +0.941] | +0.030 [0.000, +0.070] |
| E − E_hcw | −0.046 [−0.070, −0.022] | −0.010 [−0.030, 0.000] |
| E_hcw − Ehcw_k | +0.032 [+0.018, +0.047] | −0.020 [−0.050, 0.000] |
| Ehcw_k − F600 | +0.068 [+0.044, +0.092] | 0.000 |
| E_hcw − F600 | +0.100 [+0.077, +0.124] | −0.020 [−0.050, 0.000] |
| F600 − B | +0.127 [+0.104, +0.152] | 0.000 |

F600 validation curve (stream 4, `s25_val_F` log) [V]: p* 0.02 → free 1.000, Δv 0.553; 0.05 → 1.000, 0.366; 0.10 → 1.000, 0.242; 0.20 → 0.960, 0.132.

**Against the pre-stated predictions:**

| Prediction | Outcome |
|---|---|
| Fixed-τ loses to margin | Supported for HCW fixed-τ (+0.032 [0.018, 0.047]). Straight-line E (0.234) is not clearly beaten by margin (0.248); E beats E_hcw here. |
| Margin ≈ F | **Not supported:** F600 cheaper by 0.068 [0.044, 0.092] at equal pilot safety. |
| F ≤ margin | Supported at this condition. |

**Limits.** n = 100, one heterogeneity condition. E and E_hcw fall below 0.98 on stream 6 (0.970, 0.980), so part of E's cost advantage over E_hcw is bought with collisions. F600 and E were selected exactly at val 0.980 (2 failures in 100). No matched homogeneous control at the same mean σ, so "heterogeneity explains F's win" is a hypothesis. E_hcw worse than E here contradicts §6.2's dynamics-fidelity saving; the grids differ, so do not claim a direction. F600 is still about 3.4× the oracle.

### 6.4 n = 300 selection rerun [abandoned]

A rerun at n = 300 (`outputs/ladder_het_n300_r2`, 18 jobs, includes FH1/FH2) was started to select thresholds on 300 validation trials. Progress on the shared 12-thread machine was too slow (1 of 18 jobs done after about 9 h), so it was stopped; an FC (low-threshold) run was also started and stopped. **No number from either run is used.** The thresholds frozen for the final evaluation come from the n = 100 stream-4 selection in §6.3, so selection is coarse. F400 and F200 were never part of the frozen source and are excluded.

### 6.5 Final evaluation, stream 7 (run once) [V]

Thresholds frozen from §6.3 (`frozen_thresholds.json`, commit `8618a56`), then run once on 300 fresh scenarios (`freeze_final.py --execute`, `--mc-gate 6`, noise seed 12345, {5, 50} mixture, short preset). Stream 7 is spent: do not retune on it or rerun without labelling the result post hoc.

| Arm | Policy | Val free / Δv (n = 100) | Stream 7 collision-free [95% CI] | Stream 7 Δv [95% boot CI] | Δv vs val |
|---|---|---|---|---|---|
| A0 | dist_D4 | 1.000 / 1.140 | 1.000 [0.987, 1.000] | 1.150 [1.116, 1.187] | +0.009 |
| E | heuristic_tau1.2 | 0.980 / 0.199 | **0.953 [0.923, 0.972]** | 0.218 [0.201, 0.235] | +0.019 |
| E_hcw | hcw_tau1.5 | 1.000 / 0.272 | 0.990 [0.971, 0.997] | 0.279 [0.260, 0.297] | +0.007 |
| Ehcw_k | hcwm_k1 | 0.990 / 0.220 | 0.990 [0.971, 0.997] | 0.242 [0.224, 0.260] | +0.022 |
| F600 | prob_analytic_p0.15_h3 | 0.980 / 0.171 | 0.990 [0.971, 0.997] | 0.183 [0.167, 0.200] | +0.012 |
| B | prob_oracle_p0.5_h3 | 1.000 / 0.047 | 1.000 [0.987, 1.000] | 0.047 [0.040, 0.055] | 0.000 |

Paired differences on stream 7 (first minus second; positive Δv = first costs more):

| Pair | Δv | Collision-free |
|---|---|---|
| A0 − E | +0.932 [+0.899, +0.967] | +0.047 [+0.023, +0.070] |
| E − E_hcw | −0.061 [−0.076, −0.046] | −0.037 [−0.060, −0.017] |
| E_hcw − Ehcw_k | +0.036 [+0.027, +0.046] | 0.000 [0.000, 0.000] |
| Ehcw_k − F600 | +0.059 [+0.047, +0.071] | 0.000 [−0.010, +0.010] |
| E_hcw − F600 | +0.096 [+0.083, +0.108] | 0.000 [−0.010, +0.010] |
| F600 − B | +0.136 [+0.122, +0.150] | −0.010 [−0.020, 0.000] |

Against the pre-stated predictions:

| Prediction | Outcome on stream 7 |
|---|---|
| Fixed-τ loses to margin | **Supported at matched safety**: E_hcw − Ehcw_k = +0.036 [0.027, 0.046], collision outcomes identical (0.990 both). |
| Margin ≈ F | **Not supported**: F600 is cheaper by 0.059 [0.047, 0.071] at equal safety (0.990 both). |
| F ≤ margin | Supported. |

Limits. (1) One heterogeneity condition; no matched homogeneous control, so "heterogeneity explains F's advantage" remains a hypothesis. (2) Thresholds were selected on 100 validation trials; bootstrap intervals hold them fixed and omit selection noise. (3) E fell below the 0.98 target (0.953 [0.923, 0.972]), so E being cheaper than E_hcw is not a matched comparison and should not be stated as E beating E_hcw. (4) F's grid has more thresholds (9) than the HCW arms (7) and E (5); the validation-to-final Δv rise is +0.012 for F against +0.022 for Ehcw_k and +0.007 for E_hcw, which gives no sign of extra winner's-curse for F (suggestive only). (5) F600 is still about 3.9 times the oracle's Δv (0.183 vs 0.047). (6) F400 and F200 were not evaluated.

### 6.6 ESA CDM study (retrospective; Kelvins Collision Avoidance Challenge) [V]

Data: train 13,154 events / 162,634 CDMs (2.77% final-risk > −6); test 2,167 events, 150 high-risk by `true_risk` > −6 (6.92%). The dataset text defines the target as the final CDM's risk; the −6 threshold is the competition convention [A: not in the dataset text]. Scripts: `esa_study.py`, `esa_followup.py`, `make_esa_figures.py`; outputs under `outputs/esa_study`. Cost model: a flag costs r, a missed high-risk event costs 1; **no fuel claim**. Test events/positives at the 2/3/5-day cuts: 2,167/150, 2,045/138, 1,753/106. The features come from the last CDM at or before the cut. The learned model is gradient boosting on last-CDM features (not a sequence model); fit/validation is a 70/30 split of train events.

- Tuned threshold on ESA's own risk (S3) has lower regret than the tuned learned model (S4) in 8 of 9 cut × r cells (e.g. 2 d, r = 0.05: −0.0107 [−0.0152, −0.0067]); the exception is 5 d, r = 0.05 (−0.0013 [−0.0041, +0.0014]). The learned model adds nothing over the risk feature, so H3 stays untested.
- Untuned S4 (flag iff p ≥ r) vs tuned S4 has no consistent sign (2 d, r = 0.05: −0.0091 [−0.0133, −0.0053]; 3 d, r = 0.2: +0.0049 [+0.0022, +0.0080]).
- S3's raw ECE and untuned regret are artefacts (10^risk ≈ 0 flags nothing); ignore them.
- Prior shift: prevalence 2.7% (train) vs 6.9% (test). S4 calibration ratio (mean p / prevalence) by chaser-covariance tertile: validation 0.83 / 0.89 / 1.08 (2 d); test raw 0.26 / 0.34 / 0.63; test prior-shift-corrected (diagnostic; uses test prevalence) 0.54 / 0.58 / 0.89 (2 d), 0.45 / 0.44 / 1.04 (3 d), 0.59 / 0.64 / 1.40 (5 d). Calibration varies with covariance size beyond the prior shift, qualitatively as in P3, but tertiles hold 7–64 positives (validation) and 26–63 (test), the ratios have no confidence intervals, and test also has covariate shift.
- Regret per tertile does not track that gradient consistently (untuned relative regret 0.17 / 0.14 / 0.32 at 2 d corrected vs 0.24 / 0.28 / 0.27 at 3 d), so "which calibration measure predicts regret" is **inconclusive** here.

### 6.6b Calibration-regret analysis (n = 100, `ladder_het`, global logit-affine distortions) [V]

`analyze_calibration_regret.py`, F600 family, cost J = L·collision rate + Δv, c_b = 0.33 m/s, selection stream 4, evaluation stream 6. With nominal p* = c_b/L: at L = 3 (p* = 0.11, tuned optimum 0.2) the calibrated-forecast regret is +0.058 [0.023, 0.088] m/s; at L = 10 (p* = 0.033, tuned optimum 0.1) it is +0.221 [0.196, 0.250], so the untuned one-shot threshold sits below the tuned optimum. With the tuned optimum as nominal, unnoticed under-confidence (a = 0.5) costs +0.177 [0.134, 0.214] and risk over-estimation (b = +1) +0.106 [0.069, 0.139] at L = 3. **Rows marked "clamped" (all L ≥ 30 under the cost nominal) are lower bounds at the grid edge, and the L ≥ 30 rows under the tuned nominal have intervals reaching +2.2; do not use either.** Global monotone distortions only.

### 6.7 Forecast skill (stream 4, 20 episodes, 5,497 correlated rows)

Event rates at H = 100/200/400/600 s: 0.5%/1.1%/2.0%/2.5%. Brier skill vs base rate:

| σ_p | H = 100 / 200 / 400 / 600 s |
|---|---|
| 5 m | 0.996 / 0.970 / 0.896 / 0.790 |
| 25 m | 0.844 / 0.733 / 0.538 / 0.440 |
| 50 m | 0.584 / 0.483 / 0.319 / 0.257 |

ECE ≤ 0.015; 20 episodes make the calibration reading suggestive only.

### 6.8 Reference policies (stream 6, 100 episodes)

| Preset | Policy | Collision-free [95% CI] | Δv / agent |
|---|---|---|---|
| short | zero thrust | 0.550 [0.452, 0.644] | 0 |
| short | random thrust | 0.730 [0.636, 0.807] | 17.5 |
| short | full burn | 0.770 [0.678, 0.842] | 30.8 |
| long | zero thrust | 0.550 [0.452, 0.644] | 0 |
| long | random thrust | 0.960 [0.902, 0.984] | 19.0 |
| long | full burn | 0.860 [0.779, 0.915] | 32.9 |

Short warning has headroom, so it is the main experiment.

### 6.9 RL planner (secondary; negative result)

Mission rewards ×0.05 raised reward-critic explained variance from about 0.00 to 0.85 (single seed, in-sample); observability, placeholder term and fuel penalty ruled out. Pilot (short, σ_p = 25, budget 5.7, 3 seeds × 300 episodes, n = 100, stream 6):

| Policy | Collision-free (mean ± sd) | Δv / agent |
|---|---|---|
| RL reactive | 0.913 ± 0.038 | 19.75 ± 3.24 |
| RL + oracle forecast | 0.910 ± 0.044 | 23.68 ± 2.75 |
| best rules (§6.2) | 0.97–0.99 | 0.1–0.4 |

RL burns roughly **50–200×** the fuel of the rules depending on which rule and noise level you compare (19.75 / 0.4 ≈ 50; 19.75 / 0.11 ≈ 180). The 10 Nov README's "about 50×" is the conservative end. The oracle forecast gave RL no benefit at this scale. Constraint met but not binding; fuel not priced; actions unbounded and clipped.

---

## 7. Claims register

**May be written (pilot-level, with caveats):**

| # | Claim | Evidence |
|---|---|---|
| C1 | Forecast-type triggers reach ≥ 0.97–1.00 collision-free at 0.06–0.4 m/s per agent on short-warning synthetic scenarios; a current-distance trigger needs 0.45–0.65 m/s (homogeneous). | §6.2 |
| C2 | Track extrapolation (E) beats a reactive trigger (A0), intervals exclude 0. | §6.2, §6.3 |
| C3 | Exact dynamics save 0.02–0.03 m/s in the **homogeneous** setting at 5 and 25 m. Not reproduced in the {5, 50} mixture. | §6.2 vs §6.3 |
| C4 | At σ_p = 5 m a probabilistic trigger is cheaper than a point trigger with exact dynamics (0.023 m/s). | §6.2 |
| C5 | In the {5, 50} mixture, F600 is cheaper than the k·σ margin by 0.068 [0.044, 0.092] at equal pilot safety (n = 100, one condition). Confirmed once on stream 7 (n = 300): 0.059 [0.047, 0.071] at equal safety (0.990 both); one condition, thresholds from an n = 100 selection. | §6.3, §6.5 |
| C6 | Perfect state knowledge gives about 0.05 m/s at every noise level; the gap to track-conditioned rules grows with noise. | §6.2 |
| C7 | Filter consistent under cooperative broadcast, inconsistent without it under correlated burns. | §6.1 |
| C8 | Analytic forecast has positive Brier skill at all horizons. | §6.7 |
| C9 | RL planner is far less fuel-efficient than the rules and did not benefit from an oracle forecast. | §6.9 |

**Do not write:**

- Calibration measure predicts Δv or regret on real data (ESA retrospective study run: inconclusive, §6.6).
- Heterogeneity is *why* F wins (no matched control).
- Multi-horizon forecasting helps (n = 300 pending; handoff reading only).
- Learned forecaster beats physics propagation (not built).
- F headroom beyond track-conditioned rules (F − B measures tracking).
- Any fuel claim from real data, safety guarantee, other constellation sizes, multiple simultaneous threats, maneuvering threats.
- New calibration or decision theory (prior art).

---

## 8. Theory (applications of existing results)

One-shot model: burn costs c_b, an unprevented event costs L, p* = c_b / L; p(X) = Pr(E | X); forecast q(X); trigger π_t = 1{q ≥ t}; J(π) = E[c_b π + L·1_E (1 − π)].

- **P1 Bayes ordering.** π* = 1{p ≥ p*} minimises J; for any π, J(π) − J(π*) = E[|L p − c_b| · 1{π ≠ π*}] ≥ 0. Proof: condition on X; burn costs c_b, waiting costs L p.
- **P2 Invariance under global monotone recalibration.** If q = g(p), g strictly increasing, then π_t with t = g(p*) equals π*, and the set of reachable policies {π_t}_t is unchanged: tuned-threshold frontiers do not depend on g; only the optimal t moves.
- **P3 Non-invariance under context-dependent calibration.** With q = g_C(p) context-wise, a single t yields effective thresholds g_C⁻¹(t); regret is E[|L p − c_b| · 1{p between p* and g_C⁻¹(t)}] > 0 unless the g_C agree at t. *The claim "zero pooled calibration error but positive regret" needs an explicit two-context example before it is stated.* [P]
- **P4 Threshold-local regret.** Regret of a mis-set threshold is local to the density of p near p*. To be written against `analyze_calibration_regret.py`; scope: global monotone distortions only, c_b = 0.33 m/s per threatened agent is an estimate, the σ = 5 optimum sits at the grid edge p = 0.5. [P]
- **P5 Value of waiting.** The optimal burn set lies inside the myopic burn set. Proof sketch: never burning is a feasible continuation, so E[V_{t+1}] ≤ L p_t; hence c_t ≤ E[V_{t+1}] implies c_t ≤ L p_t. Two-point example from the handoff: L = 10, c_t = 3, c_{t+1} = 3.5. *Illustration (not from the handoff; verify):* p_t = 0.4 with p_{t+1} ∈ {0.1, 0.7} equally likely gives the myopic rule burn (L p = 4 ≥ 3) but waiting value 0.5·min(3.5, 1) + 0.5·min(3.5, 7) = 2.25 < 3, so the optimal rule waits.

Cite and check precisely: Schervish; Rothblum–Yona (2022); Hu–Wu (FOCS 2024); Gopalan et al.; Kleinberg et al.; Zhao et al. (decision calibration); bounded-threshold proper scoring rules (arXiv 2504.04528). Statements not re-verified here. Never claim new theory.

---

## 9. Threats to validity

1. Straight-line collision courses favour straight-line rules; harder families are future work.
2. n = 100 per policy; selection on 100 validation trials; differences within about 0.02 m/s or 3 safety points should not be interpreted.
3. Stream 6 has been read repeatedly; stream 2 already used; final uses stream 7.
4. Cooperative-thrust assumption (§6.1).
5. Burn rule is full-thrust, no hysteresis, straight-line direction.
6. One heterogeneity mixture; no matched homogeneous control.
7. Selected thresholds sit on the 0.98 boundary; bootstrap CIs understate selection noise. Final thresholds were selected on 100 validation trials (the n = 300 rerun was abandoned); E fell below 0.98 on stream 7 (0.953).
8. RL arm: unbounded actions, fuel not priced, 300 episodes, in-sample EV.
9. M3 pilot predicts a different event ("separation below 300 km within 90 min"), 3–6 scenarios; supports no ranking.
10. Hygiene: tracked `.pyc` files (use explicit-path `git add`); `test_mappo_learns_to_reduce_collisions` likely degenerate.

---

## 10. Roadmap

- [x] Objective, env fixes, tracked observations, filter, labels, analytic forecaster, ladder, E_hcw control, horizon families, calibration–regret analysis, heterogeneity, margin policy; ladder_het (n = 100) read.
- [x] n = 300 selection rerun: **abandoned**; thresholds frozen from the n = 100 selection and committed (`8618a56`).
- [x] **Stream 7 final** run once at n = 300 (§6.5). Spent.
- [x] ESA profile and retrospective study (§6.6): inconclusive on calibration vs regret.
- [ ] LaTeX theory section with proofs (P3 example, P4 against the analysis code).
- [ ] Figure and table scripts; README and claims-register update; full suite; commit and push (explicit paths).

**Future work (label as such in the paper):** synthetic learned-forecaster arms, noise-mismatch, scenario audit, RL repair (bounded actions, fuel as objective), M5 safety-filter audit, scaling, harder scenario families, Space-Track history, homogeneous-σ matched control, context-dependent calibration experiments on synthetic data.

---

## 11. Paper outline and evidence map

| Section | Content | Source |
|---|---|---|
| 1 Introduction | Question; why signal form matters | §1 |
| 2 Related work | Decision-theoretic calibration; MDP maneuver timing (arXiv 2501.02667); Bayesian CAM planning (Greco et al.); conformal belief-space planning (ConstrainedZero); PID Lagrangian (Stooke et al.) | §8 and prior list; literature review incomplete |
| 3 Setup | HCW, event definition, scenarios, Δv metric | §2, §4 |
| 4 Tracking and forecasting | Kalman, NEES, analytic MC forecaster | §6.1, §6.7 |
| 5 Theory | P1–P5 | §8 |
| 6 Protocol | Streams 4/6/7, selection, paired bootstrap | §5 |
| 7 Results | Ladder, heterogeneity, margin vs F, value decomposition; ESA if completed | §6.2–§6.6 (final numbers from stream 7) |
| 8 Discussion | What extrapolation, dynamics, margin and probability each add; tracking-limited headroom | §6, §7 |
| 9 Limitations | §9 plus cut items | §9, §10 |

Figures: Δv–safety frontier per arm; value-decomposition bars A0 → E → E_hcw → Ehcw_k → F → B; F600 validation curve; NEES with/without broadcast; Brier skill vs horizon; ECE vs regret (ESA).

**Minimum submittable state:** theory P1–P3 with proofs and P5 example; ladder results on streams 4/6 (n = 300 if finished, else n = 100 labelled pilot-level); stream 7 final (done, once); limitations stated; ESA, learned forecasters and all §10 future-work items labelled as future work.

---

## 12. Reproducing

```powershell
git clone https://github.com/Sharvari226/constellAI
cd constellAI
git switch feat/noisy-tracks
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
$env:PYTHONPATH = (Get-Location).Path

python -m pytest tests/unit -q
python constellai\scripts\run_ladder.py --sigmas 25 --sigma-choices 5 50 --families A0 E E3 HM F F2 FH2 FH1 B --trials 300 --preset short --workers 10 --out-root outputs\ladder_het_n300_r2
python constellai\scripts\select_and_compare.py --root outputs\ladder_het_n300_r2 --target 0.98 --boot 2000
```

Final evaluation (after freezing): `evaluate_tracked.py --final` with the frozen thresholds, never before. Space-Track credentials, if ever used: env vars `SPACETRACK_USERNAME`, `SPACETRACK_PASSWORD`; never paste them.

---

## 13. Repository layout

```
constellai/
  orbital_mechanics/  tle, propagation, relative_motion (HCW), conjunction, synthetic, fetch, spacetrack
  graph/              conjunction-candidate graphs, screening
  models/tgnn/        TGN-lite, LSTM, GNN baselines (M3 pilot), risk adapter
  models/marl/        environments, MAPPO family, constrained_mappo, objective, joint_obs, checkpoint
  safety/             CBF filters (unaudited)
  tracking/           kalman.py, tracker.py
  forecasting/        events.py
  evaluation/         scenarios, trials, scaled_env, heuristic, triggers, stats
  scripts/            run_ladder, select_and_compare, evaluate_tracked, analyze_calibration_regret,
                      profile_esa_cdm, run_pilot, analyze_pilot (RL arms only), check_*, reference_baselines
tests/unit/
outputs/              ladder1, ladder_het, ladder_het_n300_r2 (running), pilot1; old m4*/ superseded
data/esa_cdm          gitignored
```

---

## 14. Working conventions

- Explicit-path `git add`, never `-A`. Feature branches; no direct push to `main`.
- Patches are guarded, idempotent Python (anchor-checked, compiled, per-edit report, nothing written on failure, BOM/CRLF preserved). Embedded script strings use `r'''`. New files written by Python.
- Policy names: `dist_D*`, `heuristic_tau*`, `hcw_tau*`, `hcwm_k*`, `prob_analytic_p*_h{1,2,3}`.
- Results contradicting the hypothesis are reported as such; pre-fix results never mixed into a table; every status claim re-verified before reuse.
- Gotchas: "sigma_p = 25" in `ladder_het*` is the {5, 50} mixture; `c_b = 0.33 m/s` is an estimate; gate tests need converged-filter velocity std; LF/CRLF warnings harmless; quote paths with spaces; changing `--trials` on a resumable runner mixes sample sizes.

*Licence and citation: to be added.*
