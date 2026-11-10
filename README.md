# ConstellAI

**Forecast-conditioned, fuel-aware maneuver planning for satellite constellations under uncertain tracks.**

> **Status (10 Nov 2026): research prototype; pilot-level evidence exists; the final evaluation has not been run.**
> All numbers below are from a development protocol (100 trials per policy, thresholds selected on a validation
> stream, results read on a development stream). They are suitable for drafting the paper's structure and
> arguments, **not** for headline claims. Section 7 lists exactly which claims are supported, which are not, and
> what would change them.

---

## Table of contents

1. [Aims and research question](#1-aims-and-research-question)
2. [Setting and modelling choices](#2-setting-and-modelling-choices)
3. [What is implemented](#3-what-is-implemented)
4. [Planners compared](#4-planners-compared)
5. [Experimental protocol](#5-experimental-protocol)
6. [Results to date](#6-results-to-date)
7. [Claims register](#7-claims-register)
8. [Threats to validity and known issues](#8-threats-to-validity-and-known-issues)
9. [Roadmap](#9-roadmap)
10. [Paper outline and evidence map](#10-paper-outline-and-evidence-map)
11. [Reproducing](#11-reproducing)
12. [Repository layout](#12-repository-layout)
13. [Related work](#13-related-work)
14. [Established methods vs. our contributions](#14-established-methods-vs-our-contributions)
15. [Working conventions](#15-working-conventions)

---

## 1. Aims and research question

Low Earth orbit is congested, and conjunction management combines three things: anticipating risk, deciding under
uncertainty, and conserving fuel. ConstellAI tests whether *how* a risk signal is computed changes the fuel needed to
stay safe.

> **Does calibrated, multi-horizon conjunction-risk forecasting improve the safety–fuel trade-off of
> forecast-conditioned satellite maneuver planning, compared with reactive planning under the same dynamics,
> constraints, and evaluation scenarios?**

### 1.1 How the question was refined by the evidence

A first-pass learned (RL) planner burned about 20 m/s per agent while a 40-line threshold rule achieved the same or
better safety at about 0.1–0.4 m/s. Two consequences followed:

1. In deterministic dynamics, a forecast that is a function of the observed track carries no information a learned
   policy could not compute itself, so "reactive vs. forecast" cannot mean "state vs. forecast". The meaningful
   comparison is between **trigger signals**, from the weakest to the strongest:
   current distance → straight-line miss extrapolation → exact-dynamics miss extrapolation → calibrated probability
   from a forecaster → oracle (Section 4).
2. The primary experiment is therefore a **trigger ladder** with a shared burn rule. The RL planner is retained as a
   secondary arm and reported as a negative result.

### 1.2 Hypotheses and current status

| # | Hypothesis | Status |
|---|---|---|
| H1 | Forecast-type triggers beat a purely reactive trigger on the Δv-vs-collision frontier. | **Supported (pilot).** Extrapolation (E) beats current-distance (A0) with intervals excluding zero at all noise levels. |
| H1b | Probabilistic, uncertainty-aware forecasting beats a point forecast with correct dynamics. | **Mixed.** Cheaper at low noise (σ_p = 5 m); no demonstrated difference at σ_p = 50 m; unresolved at 25 m (safety not matched). |
| H2 | Calibration matters for planning. | **Untested, and at risk.** A trigger whose threshold is tuned on validation data is invariant to any monotone miscalibration. Calibration can only matter when the decision uses absolute probabilities (Section 9, task C3). |
| H3 | A learned forecaster adds value over physics-based propagation. | **Not testable yet.** The learned forecaster (arms C/D) is not built. |
| H4 | Multi-horizon information improves the trade-off (timing of the burn). | **Untested.** The probability trigger currently watches only the 600 s horizon. |

### 1.3 Explicit non-claims

- **No formal safety guarantee** is claimed anywhere.
- **No novelty claim** is made; the literature review is incomplete (Section 13).
- Tracking noise is a **synthetic sensitivity setting**, not realistic orbit-determination uncertainty.
- Forecast calibration is **relative to the scenario distribution**, not to real-world conjunction prevalence.
- Everything here is from **synthetic, straight-line-geometry scenarios** (Section 8).

---

## 2. Setting and modelling choices

| Item | Choice |
|---|---|
| Dynamics | Linear Hill–Clohessy–Wiltshire (HCW) relative motion in the LVLH frame; reference circular orbit at 500 km altitude (WGS-72 constants) |
| Primary | Static object at the frame origin, treated as a pseudo-threat with relative state `−x_ego` |
| Agents | N = 3 controlled satellites sharing one policy; thrust limit 0.01 m/s² per axis |
| Step | dt = 10 s; up to 200 steps per episode |
| Collision | Separation < r_c = 50 m on the step grid, strict inequality (a **proxy sphere**, not a hard-body radius) |
| Fuel metric | Δv = Σ‖thrust‖·dt in m/s, reported **per agent averaged over all agents** (including agents never threatened) |
| Tracks | Kalman filter per ego–threat pair on position-only measurements with isotropic noise σ_p ∈ {5, 25, 50} m (0.1, 0.5, 1.0 r_c) |
| Neighbour thrust | **Cooperative assumption**: neighbours broadcast their applied thrust (Section 6.1) |
| Scenarios | Built backwards from a chosen time of closest approach; 15% of agents on a zero-thrust collision course (miss ≤ 40 m), the rest pass at 75–200 m; warning time to closest approach **short = 150–600 s (main)** or long = 500–1500 s |

Sanity check on the fuel scale: a 50 m lateral shift needs Δv ≈ 2d/T ≈ 0.2–0.7 m/s per threatened agent for these
warning times (back-of-envelope, straight-line, ignoring HCW coupling). With 15% of agents threatened, the oracle's
0.05 m/s per agent corresponds to about 0.3 m/s per threatened agent, consistent with that estimate.

---

## 3. What is implemented

All code is on branch `feat/noisy-tracks`; the tag `baseline-pre-objective` pins the original code
(`edcaf04`).

| Area | Files | Verification |
|---|---|---|
| Environment fixes | `models/marl/multi_agent_environment.py` | Two-phase `step()` (collisions and costs independent of agent order); fuel penalty applied once; applied thrust and true-state accessors; nominal-path mission term (legacy shell term kept as `mission_mode="primary_shell"`) |
| Kalman filter and tracked environment | `tracking/kalman.py`, `tracking/tracker.py` | 8 + 10 tests: transition matrices match the propagator; NEES consistency with a test that has power to detect inconsistency; observation contract and information parity between arms |
| Labels and analytic forecaster | `forecasting/events.py` | Labels equal a brute-force propagator and the environment's first collision step; forecast is monotone in horizon and exact in the zero-uncertainty limit; optional σ-gate (2.7× faster, identical per-trial results at n = 20) |
| Trigger ladder | `evaluation/heuristic.py`, `evaluation/triggers.py` | Direction rule shared across arms; each trigger tested in isolation and in the tracked environment |
| Constrained objective (RL arm) | `models/marl/objective.py`, `constrained_mappo.py`, `joint_obs.py`, `checkpoint.py` | Shaped proximity cost, PID Lagrangian on observed discounted cost, privileged critic, `actor_obs_dim` in checkpoints |
| Evaluation and analysis | `scripts/evaluate_tracked.py`, `run_ladder.py`, `select_and_compare.py`, `run_pilot.py`, `analyze_pilot.py` | Raw per-trial results saved; paired bootstrap over scenarios |
| Diagnostics | `check_track_consistency.py`, `check_forecast_calibration.py`, `reference_baselines.py`, `diagnose_actor_saturation.py` | Run and reviewed |

Unit suite at the last full run: **299 passed, 1 skipped**. One newer test (the σ-gate coverage test) was being
repaired at the time of writing.

---

## 4. Planners compared

All trigger-ladder arms use **one burn rule**: when triggered, burn at full thrust along `−m`, where `m` is the
predicted closest-approach vector from the track means under straight-line motion (if exactly head-on, burn
perpendicular to the relative velocity). They stop burning when the trigger clears (no hysteresis). Only the trigger
signal differs.

| Arm | Trigger signal | What it adds over the previous arm |
|---|---|---|
| **A0** | current separation < D·r_c | pure reactive |
| **E** | straight-line predicted miss < τ·r_c | extrapolation of the track |
| **E_hcw** | miss from exact HCW propagation of the track mean < τ·r_c | correct dynamics (control for E vs. F) |
| **F** | P(entry within H) ≥ p\* from the analytic Monte Carlo forecast of the track posterior | uncertainty and probability (H = 600 s so far) |
| **B** | exact event indicator (oracle) | perfect state knowledge, not a forecasting bound |
| **RL** | Constrained MAPPO on tracked observations (secondary) | learned policy, 3 seeds × 300 episodes |

**Event definition.** `E_H` = the ego–threat separation falls below r_c at some step within H seconds, with both
bodies applying zero thrust from the forecast time. It is *risk if no action is taken*, not the probability of
collision after the planner maneuvers. Horizons are {100, 200, 400, 600} s; events are nested, so forecasts are
monotone in H by construction.

**Analytic forecaster.** Sample the filter posterior `N(μ, P)`, propagate each sample with `Φ^s`, and report
first-passage frequencies (common random numbers → deterministic and monotone).

**RL objective (secondary arm).** Maximise a mission reward (deviation from the zero-thrust nominal path, capped,
plus a smoothness term, with the fuel penalty added by the scaling wrapper) subject to a budget on a shaped
proximity cost (a *proxy*, not a bound on collision probability). Mission rewards are multiplied by a fixed 0.05,
which fixed the critic (Section 6.5).

---

## 5. Experimental protocol

| Stream | Role |
|---|---|
| 1 | RL policy training |
| 3 / 5 | Forecaster training / calibration (reserved for the learned forecaster) |
| **4** | **Validation:** choose each family's threshold |
| **6** | **Development:** read selected thresholds, inspect, debug. Looked at repeatedly, so treated as a development set |
| 2 | Legacy evaluation stream. **Already used** for two non-adaptive reference policies (zero thrust, constant burn) and for legacy pre-fix diagnostics |
| **7** | **Final stream:** one-time evaluation of frozen thresholds. Never read |

Selection rule: for each family, among thresholds with validation collision-free rate ≥ 0.98, take the one with the
smallest validation mean Δv. Intervals are **paired bootstrap over scenarios** (trial *i* is the same scenario and
measurement-noise realisation for every policy). With 100 trials the selection itself is noisy; validation-to-
development regression is visible (Section 6.3). The final evaluation will use more trials and frozen thresholds.

---

## 6. Results to date

*Everything in Section 6 is pilot/development evidence unless stated otherwise.*

### 6.1 Tracking validity (NEES; target 6; 30 episodes, short preset, validation stream)

| Configuration | Primary pair | Neighbour pair |
|---|---|---|
| neighbour thrust **not** broadcast, zero thrust | 5.95–6.09 | 3.34 / 3.49 / 3.60 (σ_p = 5 / 25 / 50) |
| not broadcast, random thrust | same | 6.06 / 6.03 / 6.00 |
| not broadcast, **constant burn** | same | **35.6 / 75.6 / 101.3** |
| **thrust broadcast**, all three policies | 5.95–6.09 | **5.69–5.86** |

The non-cooperative filter is biased when every agent burns the same thrust and cannot be repaired by tuning process
noise. The main experiment therefore **assumes cooperative thrust broadcast** and must state it. The test passes by
construction in the sense that the filter shares the simulator's dynamics; the noise-mismatch experiment (Section 9,
C5) is what tests robustness.

### 6.2 The analytic forecast has skill (validation stream; 20 episodes, 5,497 correlated rows)

Event rates at H = 100 / 200 / 400 / 600 s: 0.5% / 1.1% / 2.0% / 2.5%.

| σ_p | Brier skill vs. base rate at H = 100 / 200 / 400 / 600 s |
|---|---|
| 5 m | 0.996 / 0.970 / 0.896 / 0.790 |
| 25 m | 0.844 / 0.733 / 0.538 / 0.440 |
| 50 m | 0.584 / 0.483 / 0.319 / 0.257 |

All skill confidence intervals exclude zero. ECE ≤ 0.015 (largest at σ_p = 50 m, H = 600 s), against floors of about
0.001; with 20 episodes the calibration reading is suggestive only.

### 6.3 The trigger ladder (main result)

Short warning, **σ_p ∈ {5, 25, 50} m**, 100 trials per policy. Δv is mean per agent in m/s. "val" = stream 4 (where
the threshold was chosen), "dev" = stream 6 (reported). Brackets: Wilson interval for the rate, bootstrap interval
for Δv.

**σ_p = 5 m**

| Arm | Chosen | val free / Δv | dev collision-free | dev Δv |
|---|---|---|---|---|
| A0 | D = 2 | 0.980 / 0.469 | 0.990 [0.946, 0.998] | 0.445 [0.380, 0.513] |
| E | τ = 1.05 | 1.000 / 0.119 | 1.000 [0.963, 1.000] | 0.110 [0.092, 0.129] |
| E_hcw | τ = 1.0 | 1.000 / 0.091 | 0.990 [0.946, 0.998] | 0.082 [0.068, 0.095] |
| F | p\* = 0.4 | 1.000 / 0.060 | 1.000 [0.963, 1.000] | 0.058 [0.045, 0.073] |
| B | oracle | 1.000 / 0.046 | 1.000 [0.963, 1.000] | 0.047 [0.033, 0.061] |

**σ_p = 25 m**

| Arm | Chosen | val free / Δv | dev collision-free | dev Δv |
|---|---|---|---|---|
| A0 | D = 2.5 | 1.000 / 0.629 | 1.000 [0.963, 1.000] | 0.633 [0.559, 0.708] |
| E | τ = 1.2 | 0.990 / 0.246 | 0.970 [0.915, 0.990] | 0.235 [0.204, 0.267] |
| E_hcw | τ = 1.2 | 0.990 / 0.215 | 0.970 [0.915, 0.990] | 0.211 [0.184, 0.238] |
| F | p\* = 0.3 | 0.990 / 0.107 | **0.940** [0.875, 0.972] | 0.112 [0.091, 0.133] |
| B | oracle | 1.000 / 0.049 | 1.000 [0.963, 1.000] | 0.056 [0.037, 0.080] |

**σ_p = 50 m**

| Arm | Chosen | val free / Δv | dev collision-free | dev Δv |
|---|---|---|---|---|
| A0 | D = 2.5 | 0.980 / 0.623 | 0.990 [0.946, 0.998] | 0.647 [0.575, 0.721] |
| E | τ = 1.5 | 0.980 / 0.363 | 0.950 [0.888, 0.978] | 0.427 [0.383, 0.473] |
| E_hcw | τ = 1.5 | 0.980 / 0.368 | 0.980 [0.930, 0.994] | 0.406 [0.362, 0.453] |
| F | p\* = 0.07 | 0.980 / 0.395 | 0.990 [0.946, 0.998] | 0.423 [0.381, 0.465] |
| B | oracle | 1.000 / 0.053 | 1.000 [0.963, 1.000] | 0.051 [0.037, 0.065] |

**Paired differences on the development stream** (first minus second; Δv in m/s; positive Δv means the first costs
more; collision-free differences are in rate, negative means the first is less safe):

| σ_p | A0 − E | E − E_hcw | E_hcw − F | F − B |
|---|---|---|---|---|
| 5 m, Δv | +0.336 [0.276, 0.395] | +0.028 [0.014, 0.043] | +0.023 [0.015, 0.032] | +0.012 [0.003, 0.021] |
| 5 m, safety | −0.010 [−0.030, 0.000] | +0.010 [0.000, 0.030] | −0.010 [−0.030, 0.000] | 0.000 |
| 25 m, Δv | +0.398 [0.333, 0.462] | +0.024 [0.006, 0.043] | +0.099 [0.078, 0.121] | +0.056 [0.033, 0.077] |
| 25 m, safety | +0.030 [0.000, 0.070] | 0.000 | +0.030 [−0.010, 0.080] | −0.060 [−0.110, −0.020] |
| 50 m, Δv | +0.220 [0.161, 0.279] | +0.021 [−0.013, 0.055] | −0.017 [−0.049, 0.015] | +0.373 [0.337, 0.410] |
| 50 m, safety | +0.040 [0.010, 0.080] | −0.030 [−0.070, 0.000] | −0.010 [−0.030, 0.000] | −0.010 [−0.030, 0.000] |

**Reading:**

1. **Extrapolating the track is worth a lot.** A0 costs about 4× E at σ_p = 5 m, 2.7× at 25 m and 1.5× at 50 m, with
   intervals excluding zero. (An earlier coarse A0 grid had inflated this to "3–10×"; the finer grid is used here.)
2. **Correct dynamics saves 0.02–0.03 m/s**, which is 25%, 10% and 5% of E's cost at σ_p = 5 / 25 / 50 m. The interval
   excludes zero at 5 and 25 m. A straight-line extrapolation is an approximation over 600 s because the orbit turns
   about 0.66 rad in that time.
3. **Probabilistic handling vs. a point forecast with correct dynamics:** cheaper at σ_p = 5 m (0.023 m/s, 28% of
   E_hcw's cost); **no demonstrated difference at 50 m**; at 25 m F's selected point is cheaper but 3 points less safe
   on the development stream (0.94 vs. 0.97), so the comparison is **not matched** and not interpretable as a win.
4. **F − B is the value of perfect state knowledge**, not forecasting headroom. The oracle also knows the true state,
   which no method conditioned on the noisy tracks can know. The gap grows with noise (0.012 → 0.056 → 0.373 m/s)
   because tracking error limits any track-conditioned rule.
5. **Selection noise is visible:** thresholds chosen on 100 validation trials regress on the development stream
   (F at σ_p = 25: 0.99 → 0.94; E at σ_p = 50: 0.98 → 0.95). Matched-safety comparisons need larger n or a frontier
   analysis.

Full threshold sweeps for all arms (validation and development, all three noise levels) are in
`outputs/ladder1/*/eval_*.json` with raw per-trial results.

### 6.4 Reference policies (development stream, 100 episodes, γ = 0.99)

| Preset | Policy | Collision-free [95% CI] | Agent collision rate | Δv / agent | Discounted shaped cost, mean (p90) |
|---|---|---|---|---|---|
| short | zero thrust | 0.550 [0.452, 0.644] | 0.197 | 0 | 14.3 (23.4) |
| short | random thrust | 0.730 [0.636, 0.807] | 0.107 | 17.5 | 11.0 (21.9) |
| short | full burn | 0.770 [0.678, 0.842] | 0.133 | 30.8 | 11.4 (23.3) |
| long | zero thrust | 0.550 [0.452, 0.644] | 0.197 | 0 | 9.9 (19.1) |
| long | random thrust | 0.960 [0.902, 0.984] | 0.017 | 19.0 | 3.0 (10.4) |
| long | full burn | 0.860 [0.779, 0.915] | 0.083 | 32.9 | 6.1 (15.1) |

The short preset has headroom (a random-thrust policy is 73% collision-free at 17.5 m/s); the long preset largely
does not. This is why short warning is the main experiment. Even 31 m/s of undirected thrust leaves 23% of short
trials with a collision, so coordinated, directed burns matter, not thrust magnitude.

### 6.5 The RL planner (secondary arm; negative result)

**Critic diagnosis (single seed, in-sample explained variance).** Multiplying mission rewards by 0.05 raised the
reward-critic explained variance from about 0.00 to 0.85 (last 10 episodes), with the same code. The earlier
suspects were ruled out: observability of the slot error (added to the observation, no effect on its own), the
placeholder mission term (replaced, no effect on its own), and the fuel penalty (removing it gave 0.67). Cost-critic
EV is about 0.64 with tracked observations.

**Policy pilot** (short preset, σ_p = 25 m, budget 5.7, 3 seeds × 300 episodes, 100 trials, development stream):

| Policy | Collision-free (mean ± sd over seeds) | Δv / agent | Saturation |
|---|---|---|---|
| RL reactive | 0.913 ± 0.038 | 19.75 ± 3.24 | 0.544 |
| RL + oracle forecast | 0.910 ± 0.044 | 23.68 ± 2.75 | 0.679 |
| best rule (E_hcw / F, §6.3) | 0.97–0.99 | **0.1–0.4** | n/a |

Paired RL reactive − oracle: collision-free +0.003 [−0.073, +0.080]; Δv −3.9 [−7.9, 0.0]. The learned planner is
about **50× less fuel-efficient** than the rules, and the oracle forecast gave it no benefit at this training scale.
The PID constraint is met (observed discounted cost 1.5–4.6 against a budget of 5.7), but the constraint is not
binding and fuel is not priced, so the policy burns continuously. In seed 0, 90% of steps have a mean action beyond
the thrust limit that the environment clips (axis drift of −1.74), while seed 1 stays mostly inside the limit and
still burns about 19 m/s. Hypotheses to test before any RL claim: bounded actions (tanh mean), fuel as the optimised
quantity with risk as the constraint, longer training.

### 6.6 M3 pilot (an older task; a different event)

M3 predicts "separation below 300 km within 90 minutes", not the planning event. Rerun on the current code
(LSTM and GNN: 6 test scenarios at a 2-minute step; TGN-lite: 3 test scenarios at a 5-minute step; 3 seeds):

| Model | AP (seed range) | Precision | Recall |
|---|---|---|---|
| LSTM | 0.756 (0.66–0.84) | 0.760 | 0.864 |
| Static GNN | 0.183 | 0.162 | 0.773 |
| TGN-lite (point) | 0.246 (0.18–0.31) | 0.047 | 0.711 |
| TGN-lite (Beta head) | 0.014, ECE 0.334 | | |

The effective sample is 3–6 scenarios and the comparison is unmatched, so this supports no ranking. TGN-lite does not
read event timestamps, so continuous-time claims are unsupported. It is **not** the forecaster for the central
experiment.

---

## 7. Claims register

**Claims the evidence supports now (state as pilot-level, with the stated caveats):**

| # | Claim | Evidence |
|---|---|---|
| C1 | Forecast-type triggers reach ≥ 0.97–1.00 collision-free at 0.06–0.4 m/s per agent on the short-warning synthetic scenarios, where a current-distance trigger needs 0.45–0.65 m/s for similar safety. | §6.3 |
| C2 | Extrapolating the track (E) beats a purely reactive trigger (A0); intervals exclude zero at all three noise levels. | §6.3 |
| C3 | Using exact dynamics for the extrapolation saves 0.02–0.03 m/s (interval excludes zero at σ_p = 5 and 25 m). | §6.3 |
| C4 | At low noise (σ_p = 5 m), a probabilistic trigger is cheaper than a point trigger with exact dynamics (0.023 m/s). | §6.3 |
| C5 | Perfect state knowledge would cut Δv to about 0.05 m/s at every noise level; the gap to the best track-conditioned rule grows with noise. | §6.3 |
| C6 | The filter is consistent under cooperative thrust broadcast and inconsistent without it under correlated burns. | §6.1 |
| C7 | The analytic forecast has positive Brier skill at all horizons and noise levels. | §6.2 |
| C8 | The RL planner implemented here is far less fuel-efficient than the rules and did not benefit from an oracle forecast. | §6.5 |

**Claims the evidence does not support (do not write them):**

- Calibration improves planning (untested; invariance under tuned thresholds is a threat).
- Multi-horizon forecasting improves planning (untested).
- A learned forecaster adds value over physics-based propagation (not built).
- A probabilistic forecast helps at high noise (no difference demonstrated at σ_p = 50 m) or at 25 m (unmatched safety).
- Forecasting has headroom beyond the track-conditioned rules (F − B measures tracking, not forecasting).
- Any statement about safety guarantees, other constellation sizes, multiple simultaneous threats, maneuvering threats,
  or real orbital data.
- Any ranking of M3 models.

---

## 8. Threats to validity and known issues

1. **Scenario generator.** Straight-line collision courses favour straight-line extrapolation rules. Harder families
   (simultaneous threats, a maneuvering threat, longer horizons) are needed before any claim travels.
2. **Sample size and selection.** 100 trials per policy; thresholds selected on 100 validation trials. Differences
   within about 0.02 m/s or 3 safety points should not be interpreted.
3. **The development stream has been looked at repeatedly.** Stream 2 was already used for reference policies and
   legacy diagnostics; the final evaluation uses a fresh stream (7).
4. **Cooperative-thrust assumption.** The non-cooperative filter is inconsistent (§6.1).
5. **Single-horizon probability trigger.** The multi-horizon part of the research question is untested.
6. **Burn rule.** Full-thrust, no hysteresis, straight-line direction. Burn direction and timing optimisation are not
   explored.
7. **RL arm.** Unbounded actions with an environment clip; fuel not priced; 300 episodes only; mission reward scaled
   by a fixed constant (a hyperparameter, not a principled normalisation); explained variance reported in-sample.
8. **M3 forecaster mismatch.** Different event, one horizon, no timestamp use, collapsed uncertainty head,
   unverified calibration.
9. **Safety filter (M5) unaudited.** No safety claim is supportable yet.
10. **Hygiene.** About 50 `.pyc` files are tracked in git; `test_mappo_learns_to_reduce_collisions` starts the
    agents inside the collision radius and is likely degenerate; some docstrings are stale.

---

## 9. Roadmap

**Done:** objective and environment fixes; tracked-observation path; filter validation; labels and analytic
forecaster; trigger ladder with exact-dynamics control; first pilot ladder; RL pilot and its diagnosis.

**Next (in order):**

- [ ] **P1. Horizon-timing families.** Triggers on the 400 s and 200 s horizons (F400, F200), each with the same
      tuning budget as the others, to test the multi-horizon part of the question.
- [ ] **P2. Frozen protocol at larger n.** Thresholds chosen on stream 4 at about 300 trials; one evaluation on the
      fresh final stream 7 with frozen thresholds; matched-safety comparison or frontier analysis.
- [ ] **C3. Cost-ratio trigger.** Burn iff p ≥ c/(c + L) (fuel cost c, collision cost L), untuned. The only setting in
      which calibration can matter.
- [ ] **C1–C6. Learned forecaster v2** (interval-hazard survival loss, MLP first, shared monotone calibration,
      validated against the base rate, the analytic forecast and an MLP); arms C and D; noise-mismatch experiment.
- [ ] **S1. Harder scenario families** and a robustness check on a differently shaped sampler.
- [ ] **R1. RL arm repair (optional):** bounded actions, fuel as the objective with risk as the constraint, longer
      training; report either way.
- [ ] **M5.** Audit of the safety filter before any safety statement.
- [ ] **L1.** Complete the literature review; **P.** write the paper from Section 10.

---

## 10. Paper outline and evidence map

| Section | Content | Source in this repository |
|---|---|---|
| 1 Introduction | Congested LEO; question: does the *form* of the risk signal change the fuel needed? | §1 |
| 2 Related work | MDP maneuver timing under updates; Bayesian CAM planning; conformal belief-space planning; PID Lagrangian | §13 (to be completed) |
| 3 Problem and simulator | HCW relative dynamics; event definition; scenario generator; fuel metric | §2, §4 |
| 4 Tracking and forecasting | Kalman tracks; NEES validation; analytic Monte Carlo first-passage forecast; labels verified against the environment | §6.1, §6.2; `tests/unit/test_kalman.py`, `test_forecast_events.py` |
| 5 Planners | Trigger ladder with shared burn rule; oracle; RL arm | §4 |
| 6 Protocol | Streams, selection on validation, paired bootstrap, frozen final evaluation | §5 |
| 7 Results | Ladder tables and paired differences; value decomposition; RL negative result | §6.3–§6.5 (final numbers from stream 7) |
| 8 Discussion | What extrapolation, dynamics and probability each contribute; tracking-limited headroom; calibration and decision rules | §6.3, §7 |
| 9 Limitations | Synthetic straight-line scenarios, cooperative assumption, sample size, RL arm | §8 |
| 10 Conclusion | Only claims from §7, upgraded or removed after the frozen evaluation | §7 |

Suggested figures: (1) Δv vs. collision-free frontier per arm per noise level (all thresholds); (2) bar chart of the
value decomposition A0 → E → E_hcw → F → B per noise level; (3) NEES with and without thrust broadcast; (4) Brier skill
vs. horizon; (5) RL vs. rules on the same axes.

---

## 11. Reproducing

Python 3.13 on Windows (PowerShell); NumPy and PyTorch (see `pyproject.toml`, not reviewed here).

```powershell
git clone https://github.com/Sharvari226/constellAI
cd constellAI
git switch feat/noisy-tracks
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
$env:PYTHONPATH = (Get-Location).Path

python -m pytest tests/unit -q                                           # unit suite
python constellai\scripts\check_track_consistency.py --coop --accel-std 0.0   # NEES with thrust broadcast
python constellai\scripts\check_forecast_calibration.py                  # analytic forecast skill
python constellai\scripts\reference_baselines.py                         # reference policies
python constellai\scripts\run_ladder.py --out-root outputs\ladder1 --workers 4 --mc-gate 6
python constellai\scripts\select_and_compare.py --root outputs\ladder1 --target 0.98
```

The final evaluation (not yet run) must use the fresh stream with `--final` after thresholds are frozen.

---

## 12. Repository layout

```
constellai/
  orbital_mechanics/   tle, propagation, relative_motion (HCW), conjunction, synthetic, fetch, spacetrack
  graph/               conjunction-candidate graphs and screening
  models/tgnn/         TGN-lite, LSTM and GNN baselines (M3 pilot), risk adapter
  models/marl/         environments, MAPPO family, constrained_mappo, objective.py, joint_obs.py, checkpoint.py
  safety/              CBF filters (unaudited)
  tracking/            kalman.py, tracker.py
  forecasting/         events.py (labels, analytic Monte Carlo forecaster, optional sigma gate)
  evaluation/          scenarios.py, trials.py, scaled_env.py, heuristic.py, triggers.py, stats.py, ...
  scripts/             run_ladder.py, select_and_compare.py, evaluate_tracked.py, train_tracked.py, run_pilot.py,
                       analyze_pilot.py, check_*.py, reference_baselines.py, diagnose_actor_saturation.py
tests/unit/            unit and contract tests
outputs/               ladder1/ (trigger ladder), pilot1/ (RL pilot), m3_rerun/ (M3 pilot log); older m4/ and
                       m4_scaled/ predate the environment fix and are superseded
```

---

## 13. Related work

Status: **preliminary**; the following were located and read at abstract level only.

- *Markov Decision Processes for Satellite Maneuver Planning and Collision Avoidance* (arXiv:2501.02667): how far in
  advance to maneuver to trade fuel against collision probability under information updates. Closest to our
  safety–fuel question; it concerns the **timing** trade-off that H4 targets.
- Greco et al., *A robust Bayesian agent for optimal collision avoidance manoeuvre planning* (Strathclyde, 2021):
  particle filtering plus an agent that classifies risk events and allocates manoeuvres under uncertainty.
- *ConstrainedZero* (IJCAI 2024): belief-state planning with adaptive conformal inference on a failure-probability
  threshold. Not satellite-specific; relevant to calibrated constraint handling.
- Stooke et al., *Responsive Safety in Reinforcement Learning by PID Lagrangian Methods* (ICML 2020): basis of the
  multiplier update (cited from memory, not re-verified).

Sources suggested by the team's reviewer (a NASA conjunction-assessment AI/ML compendium, a 2025 high-covariance
conjunction-prediction paper, a 2026 belief-space planning paper) could **not** be confirmed and must not be cited
until located.

---

## 14. Established methods vs. our contributions

| Component | Status |
|---|---|
| HCW dynamics; Kalman filtering; PPO/MAPPO; Lagrangian constrained RL; PID multiplier; CBF filters | Established |
| TGN, LSTM, GNN risk classifiers | Established |
| Threshold rules on miss-distance and risk probabilities | Standard operational practice in some form |
| Trigger ladder with a shared burn rule, the exact-dynamics control, and the validation-then-frozen protocol | Our experimental design |
| Decomposition of the value of forecasting into extrapolation, dynamics fidelity, probabilistic handling, and perfect state knowledge | **The intended contribution; novelty not established** |

---

## 15. Working conventions

- Work on feature branches; do not push to `main` directly.
- Add files by explicit path. Never `git add .` or `git add -A` (tracked `.pyc` files change on every run).
- Every experiment states what is tested, the controlled variables, the metrics, what would support or falsify the
  hypothesis, and what remains unjustified.
- Results contradicting the hypothesis are reported as such. Results from before the environment fix are never mixed
  into a table.
- Any status claim here is re-verified against code, tests and logs before it is reused in the paper.

*Licence and citation information: to be added.*
